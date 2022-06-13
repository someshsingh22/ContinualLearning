from cl.models.bert import FastBertModel
from cl.models.gpt import FastGPT2DistilModel
from cl.models.model import FastLearner, SlowLearner

FastModel = {
    "bert-base-uncased": FastBertModel,
    "distilgpt2": FastGPT2DistilModel,
}
