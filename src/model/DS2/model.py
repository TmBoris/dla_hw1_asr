import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .convolution import MaskCNN, get_new_sequence_lengths
from .rnn import BNReluRNN


class DeepSpeech2(nn.Module):
    """
    Args:
        input_dim (int): dimension of input vector
        n_tokens (int): number of classfication
        rnn_type (str, optional): type of RNN cell (default: gru)
        num_rnn_layers (int, optional): number of recurrent layers (default: 5)
        rnn_hidden_dim (int): the number of features in the hidden state `h`
        dropout_p (float, optional): dropout probability (default: 0.1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        activation (str): type of activation function (default: hardtanh)
    """
    def __init__(
            self,
            input_dim: int,
            n_tokens: int,
            rnn_type='gru',
            num_rnn_layers: int = 5,
            rnn_hidden_dim: int = 1024,
            dropout_p: float = 0.1,
            bidirectional: bool = True,
    ):
        super().__init__()
        
        in_channels = 1
        out_channels = 32

        self.conv = MaskCNN(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(out_channels),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(out_channels),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        self.rnn_layers = nn.ModuleList()
        rnn_input_size = 128 # n_mels
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= out_channels # because of view after conv

        for idx in range(num_rnn_layers):
            self.rnn_layers.append(
                BNReluRNN(
                    input_size=rnn_input_size if idx == 0 else rnn_hidden_dim,
                    hidden_state_dim=rnn_hidden_dim,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                    dropout_p=dropout_p,
                )
            )

        self.fc = nn.Sequential(
            nn.LayerNorm(rnn_hidden_dim),
            nn.Linear(rnn_hidden_dim, n_tokens, bias=False),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        # start spec shape (batch_size, n_mels, time)
        x, x_len = self.conv(spectrogram.unsqueeze(1), spectrogram_length) # (batch_size, channels, n_mels, time)

        batch_size, channels, n_mels, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time, channels * n_mels) # (bs, time, channels * n_mels)

        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x, x_len)

        # after rnn shape (bs, time, rnn_hidden_dim)

        log_probs = self.fc(x).log_softmax(dim=-1)
    
        return {"log_probs": log_probs, "log_probs_length": x_len}
    
    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

