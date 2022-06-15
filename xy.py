import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, BertForMaskedLM, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')
class Classifier(nn.Module):
    def __init__(self, h,out):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(h,out)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, x= self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        x = self.linear(x)
        x = self.relu(x)

        return x