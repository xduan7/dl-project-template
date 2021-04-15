import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding module in "Attention Is All You Need".

    This class instance creates a positional encoding matrix that
    contains the information about the relative or absolute position
    of the tokens in the sequence. The positional encodings have the
    same dimension as the embeddings, so that the two can be summed.
    Both sine and cosine functions of different frequencies are used
    for the positional encoding matrix.
    .. math::
        \text{PosEnc}(pos, 2i) = sin(pos/10000^(2i/emb_dim))
        \text{PosEnc}(pos, 2i+1) = cos(pos/10000^(2i/emb_dim))
        \text{where pos is the word position and i is the embed idx)

    References:
        - https://arxiv.org/abs/1706.03762
        - https://pytorch.org/tutorials/beginner/transformer_tutorial
        .html

    Args:
        seq_len: The length of the input sequence.
        emb_dim: The dimension of embedding, which is defined by the
            input (seq_len*emb_dim) of the learning model.
        emb_scale: The scaling factor for input vectors (not
            positional encoding) to prevent them from diminishing
            after adding positional encodings.
        dropout_rate: The rate of dropout after positional encoding
            is applied to the scaled input vectors.

    """

    def __init__(
            self,
            seq_len: int,
            emb_dim: int,
            emb_scale: float = 1.0,
            dropout_rate: float = 0.1,
    ):
        super(PositionalEncoding, self).__init__()
        self._emb_scale = emb_scale
        self._dropout = nn.Dropout(p=dropout_rate)

        positional_encoding = torch.zeros(seq_len, emb_dim)
        _positions = torch.arange(0, seq_len, dtype=torch.float)
        _positions = _positions.unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float()
            * (-math.log(10000.0) / emb_dim)
        )
        positional_encoding[:, 0::2] = torch.sin(_positions * div_term)
        positional_encoding[:, 1::2] = torch.cos(_positions * div_term)
        positional_encoding = \
            positional_encoding.unsqueeze(0).transpose(0, 1)

        # Uses `register_buffer` instead of `nn.Parameter()` so that
        # the positional encoding matrix is not learnable/trainable
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """Add the positional encoding information to a scaled batch
        of input sequences, and return after applying the dropout.

        """
        x = x * self._emb_scale + self.positional_encoding[:x.size(0), :]
        return self.dropout(x)
