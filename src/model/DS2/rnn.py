
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BNReluRNN(nn.Module):
    """
    Recurrent neural network with batch normalization layer & ReLU activation function.

    Args:
        input_size (int): size of input
        hidden_state_dim (int): the number of features in the hidden state `h`
        rnn_type (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        dropout_p (float, optional): dropout probability (default: 0.1)

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs
        - **outputs**: Tensor produced by the BNReluRNN module
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_size: int,
            hidden_state_dim: int = 1024,
            rnn_type: str = 'gru',
            bidirectional: bool = True,
            dropout_p: float = 0.1,
    ):
        super().__init__()
        
        self.activation = nn.Hardtanh(0, 20, inplace=True)
        self.bidirectional = bidirectional
        self.hidden_state_dim = hidden_state_dim
        # print('input_size = ', input_size)
        self.batch_norm = nn.BatchNorm1d(input_size)
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_state_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        total_length = inputs.size(1)
        x = inputs.transpose(1, 2)
        # print('shape before bn', x.shape)
        x = self.activation(self.batch_norm(x))
        x = x.transpose(1, 2)

        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        x, hidden_states = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=total_length, batch_first=True)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)

        return x
