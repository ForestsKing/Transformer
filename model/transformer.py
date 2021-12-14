from torch import nn

from model.decoder import Decoder
from model.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, d_k=64, d_v=64, d_ff=2048, d_model=512, n_heads=8, n_layers=6, tgt_vocab_size=9,
                 src_vocab_size=6):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_k, d_v, d_ff, d_model, n_layers, n_heads, src_vocab_size)
        self.decoder = Decoder(d_k, d_v, d_ff, d_model, n_layers, n_heads, tgt_vocab_size)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model]
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1))
