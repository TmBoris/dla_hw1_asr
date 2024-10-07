from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, normalized_text: List[str], **kwargs
    ):
        wers = []
        lengths = log_probs_length.detach().numpy()
        for target_text, log_prob, length in zip(normalized_text, log_probs, lengths):
            pred_text = self.text_encoder.argmax_ctc_decode(log_prob[:length, :])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, normalized_text: List[str], **kwargs
    ):
        wers = []
        lengths = log_probs_length.detach().numpy()
        for target_text, log_prob, length in zip(normalized_text, log_probs, lengths):
            pred_text = self.text_encoder.ctc_beam_search(log_prob[:length, :])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
    

class BsLmWERMetric(BaseMetric):
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, normalized_text: List[str], **kwargs
    ):
        pred_texts = self.text_encoder.lib_lm_beam_search(log_probs, log_probs_length)
        wers = [calc_wer(target_text, pred_text) for target_text, pred_text in zip (normalized_text, pred_texts)]
        return sum(wers) / len(wers)
