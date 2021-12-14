from torch import nn

from model.attention import MultiHeadAttention
from model.feed_forward import PoswiseFeedForwardNet
from model.positional_encoding import PositionalEncoding
from utils.mask import get_attn_pad_mask


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_ff, d_model, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.enc_self_attn(enc_inputs,
                                         enc_inputs,
                                         enc_inputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, d_k, d_v, d_ff, d_model, n_layers, n_heads, src_vocab_size):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_k, d_v, d_ff, d_model, n_heads) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs)  # [batch_size, src_len, d_model]

        # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model]
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs
