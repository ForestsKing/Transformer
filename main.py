import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data.dataset import MyDataSet
from model.transformer import Transformer
from utils.make_data import makedata
from utils.setseed import set_seed

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

if __name__ == '__main__':
    set_seed(2021)
    # S: Symbol that shows starting of decoding input
    # E: Symbol that shows starting of decoding output
    # P: Symbol that will fill in blank sequence if current batch data size is short than time steps
    sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
    ]

    src_len = 5  # enc_input max sequence length
    tgt_len = 6  # dec_input(=dec_output) max sequence length

    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
    src_vocab_size = len(src_vocab)

    idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    enc_inputs, dec_inputs, dec_outputs = makedata(sentences, src_vocab, tgt_vocab)
    dataset = MyDataSet(enc_inputs, dec_inputs, dec_outputs)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = Transformer(d_k=d_k, d_v=d_v, d_ff=d_ff, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        tgt_vocab_size=tgt_vocab_size,
                        src_vocab_size=src_vocab_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(1000):
        for enc_inputs, dec_inputs, dec_outputs in dataloader:
            outputs = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model, './log/Transformer.pkl')
    model = torch.load('./log/Transformer.pkl')

    testloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for s, (enc_inputs, _, _) in enumerate(testloader):
        start = torch.LongTensor([[tgt_vocab["S"]]])
        outputs = torch.LongTensor([[]])
        ifcontinue = True
        while ifcontinue:
            inputs = torch.cat([start, outputs], dim=1)
            outputs = model(enc_inputs, inputs)
            outputs = outputs.reshape(1, -1, outputs.size(-1)).argmax(dim=-1)

            ifcontinue = (outputs[0][-1] != torch.LongTensor([[tgt_vocab["E"]]]))

        print(sentences[s][0], ' -> ', ' '.join([idx2word[int(i)] for i in outputs[0][:-1]]))
