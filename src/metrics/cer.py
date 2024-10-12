from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, normalized_text: List[str], **kwargs
    ):
        cers = []
        predictions = log_probs.detach().cpu()
        lengths = log_probs_length.detach().cpu().numpy()
        for target_text, log_prob, length in zip(normalized_text, predictions, lengths):
            pred_text = self.text_encoder.argmax_ctc_decode(log_prob[:length, :])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, normalized_text: List[str], **kwargs
    ):
        cers = []
        predictions = log_probs.detach().cpu()
        lengths = log_probs_length.detach().cpu().numpy()
        for target_text, log_prob, length in zip(normalized_text, predictions, lengths):
            pred_text = self.text_encoder.ctc_beam_search(log_prob[:length, :])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
    

class BsLmCERMetric(BaseMetric):
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, normalized_text: List[str], **kwargs
    ):
        predictions = log_probs.detach().cpu()
        lengths = log_probs_length.detach().cpu().numpy()
        pred_texts = self.text_encoder.lib_lm_beam_search(predictions, lengths)
        cers = [calc_cer(target_text, pred_text) for target_text, pred_text in zip (normalized_text, pred_texts)]
        print('during cer inference info')
        print('\t cers', cers)
        print('\t pred_texts', pred_texts)
        print('\t target_texts', normalized_text)

        return sum(cers) / len(cers)
