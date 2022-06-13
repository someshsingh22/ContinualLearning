import torch.nn as nn
from transformers import AutoModelForMaskedLM

from cl.models import FastModel


class SlowLearner(nn.Module):
    def __init__(self, h, out, model_name):
        super(SlowLearner, self).__init__()
        self.h = h
        self.out = out
        self.model_name = model_name

        self.lm = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, h, x = self.lm(input_ids=input_id, attention_mask=mask, return_dict=False)
        x = self.relu(x)
        return x, h


class FastLearner(nn.Module):
    def __init__(self, h, out, model_name):
        super(FastLearner, self).__init__()
        self.h = h
        self.out = out
        self.model_name = model_name

        self.lm = FastModel[self.model_name].from_pretrained(self.model_name)
        self.linear = nn.Linear(h, out)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, x = self.lm(input_ids=input_id, attention_mask=mask, return_dict=False)
        x = self.linear(x)
        x = self.relu(x)

        return x
