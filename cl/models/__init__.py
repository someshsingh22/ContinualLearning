from cl.models.bert import FastBertForSequenceClassification
from cl.models.gpt import FastGPT2DistilForSequenceClassification

FastModel = {
    "bert-base-uncased": FastBertForSequenceClassification,
    "distilgpt2": FastGPT2DistilForSequenceClassification,
}
from cl.models.model import DualNet