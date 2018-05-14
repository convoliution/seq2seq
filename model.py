import torch
import torch.nn as nn
from torch.nn import functional as func


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        raise NotImplementedError()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        raise NotImplementedError()
