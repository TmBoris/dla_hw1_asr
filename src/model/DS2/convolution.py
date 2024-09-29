import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


def get_new_sequence_lengths(module: nn.Module, seq_lengths: Tensor) -> Tensor:
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            return seq_lengths.int() + 1

        return seq_lengths.int()


class MaskCNN(nn.Module):
    """
    Заполняем нулями пространство паддинга после применения свертки
    """
    def __init__(self, sequential: nn.Sequential) -> None:
        super(MaskCNN, self).__init__()
        self.sequential = sequential

    def forward(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output = None

        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)

            if output.is_cuda:
                mask = mask.cuda()

            seq_lengths = get_new_sequence_lengths(module, seq_lengths)

            for idx, length in enumerate(seq_lengths):
                length = length.item()

                if (mask[idx].size(2) - length) > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths

