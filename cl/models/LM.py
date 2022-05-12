import imp

import torch, transformers
import torch.nn as nn

class LM(nn.Module):
    '''
    Language model with given nu
    '''
    def __init__(self, num_heads, num_layers, hidden_size, vocab_size, dropout):

