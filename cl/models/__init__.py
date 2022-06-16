from cl.models.bert import FastBertForSequenceClassification
from cl.models.gpt import FastGPT2ForSequenceClassification
from cl.models.model import DualNet

FastModel = {
    "bert-base-uncased": FastBertForSequenceClassification,
    "distilgpt2": FastGPT2ForSequenceClassification,
}
