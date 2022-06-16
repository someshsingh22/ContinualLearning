from cl.models.bert import FastBertForSequenceClassification
from cl.models.gpt import FastGPT2ForSequenceClassification

FastModel = {
    "bert-base-uncased": FastBertForSequenceClassification,
    "distilgpt2": FastGPT2ForSequenceClassification,
}
from cl.models.model import DualNet
