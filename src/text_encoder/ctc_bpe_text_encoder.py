import re
from string import ascii_lowercase
from pyctcdecode import build_ctcdecoder
import multiprocessing
from collections import defaultdict
from tokenizers import Tokenizer
import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCBPETextEncoder:
    EMPTY_TOK = ""

    def __init__(self, lm_path, vocab_path, beam_size, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        tokenizer = Tokenizer.from_file("bpe_tokenizer.json")
        self.vocab = [self.EMPTY_TOK] + list(dict(sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])).keys())

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.beam_size = beam_size

        if lm_path is not None:
            assert vocab_path is not None, "vocab_path is None"
            with open(vocab_path, 'r') as file:
                unigrams = [x.strip() for x in file.readlines()]
            # big letters, because have big letters in unigrams
            self.decoder = build_ctcdecoder([w.upper() for w in self.vocab], kenlm_model_path=lm_path, unigrams=unigrams)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def _ctc_decode(self, inds) -> str:
        result_string = ''
        last_is_empty = False
        for ind in inds:
            char = self.ind2char[ind]

            if char == self.EMPTY_TOK:
                last_is_empty = True
                continue

            if len(result_string) == 0 or \
                (len(result_string) > 0 and (char != result_string[-1] or last_is_empty)):
                result_string += char

            last_is_empty = False

        return result_string
    
    def _expand_and_merge_path(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp

    def _truncate_paths(self, dp):
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:self.beam_size])

    def ctc_beam_search(self, probs, **other):
        dp = {
            ('', self.EMPTY_TOK): 1.0,
        }
        for prob in torch.exp(probs):
            dp = self._expand_and_merge_path(dp, prob)
            dp = self._truncate_paths(dp)
        # хотим объединить вероятности одинаковых префиксов
        final_probs = defaultdict(float)
        max_prob = -1
        best_prefics = ''
        for (prefix, _), proba in dp.items():
            assert proba >= 0
            final_probs[prefix] += proba
            if final_probs[prefix] > max_prob:
                max_prob = final_probs[prefix]
                best_prefics = prefix

        return best_prefics
    
    def argmax_ctc_decode(self, probs, **other):
        """
        :param: probs - верояности уже обрезанные по длине. shape = [length, vocab_size]
        """
        return self._ctc_decode(torch.argmax(probs.cpu(), dim=-1).numpy())

    def lib_lm_beam_search(self, probs, probs_lengths, **other):
        """
        :param: probs - батч предсказаний. shape = [bs, length, vocab_size]
        """
        if len(probs.shape) == 2:
            probs = probs.unsqueeze(0) # because don't have batch on inference

        probs = [probs[i, :probs_lengths[i], :].numpy() for i in range(probs_lengths.shape[0])]

        with multiprocessing.get_context("fork").Pool() as pool:
            texts = self.decoder.decode_batch(pool, probs, beam_width=self.beam_size)
        return [w.lower().strip() for w in texts]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
