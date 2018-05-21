import torch
import torch.nn as nn
from torch.nn import functional as func


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Encoder, self).__init__()

        self.hidden = torch.zeros(2, 1, hidden_size)

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input_seq):
        output              = self.embedding(input_seq)
        output, self.hidden = self.gru(output, self.hidden)

        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()
