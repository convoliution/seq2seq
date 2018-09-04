import torch

from utils import *
from model import *


vocab = Vocabulary(corpus="This is the entire vocabulary. Yep, everything. All the words. Yeah, that is it.")
input_seq = vocab.sequence("This is it.".split())
print("Input Sequence: {}".format(input_seq))

encoder = Encoder(vocab_size=vocab.size, hidden_size=128)
output = encoder.forward(torch.tensor(input_seq).unsqueeze(0))
