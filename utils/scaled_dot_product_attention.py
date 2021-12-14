import numpy as np
import torch
from torch import nn


def ScaledDotProductAttention(Q, K, V, mask):
    """
    Q: [batch_size, n_heads, len_q, d_k]
    K: [batch_size, n_heads, len_k, d_k]
    V: [batch_size, n_heads, len_v(=len_k), d_v]
    attn_mask: [batch_size, n_heads, seq_len, seq_len]
    """
    d_k = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
    scores.masked_fill_(mask, -1e9)  # Fills elements of self tensor with value where mask is True.
    attn = nn.Softmax(dim=-1)(scores)
    context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
    return context
