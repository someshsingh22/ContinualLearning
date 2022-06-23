# Save request json output
import json
import logging
import requests
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# from nltk.tokenize import word_tokenize
from main import MTL

def get_vocab(self): 
    final_vocab = []
    vocab = set(' '.join((MTL.dataset['train']['text'])).split(' '))
    for word in vocab:
        if word not in stopwords.word():
          final_vocab.append(word)  

    raise NotImplementedError

def save_concept(word):
    try:
        request = requests.get("http://api.conceptnet.io/c/en/{word}}")
        path = f"./data/concepts/{word}.json"

        with open(path, "w") as f:
            json.dump(request.json(), f, indent=4)
    except Exception as E:
        logging.error("Error saving request for word {}, {}".format(word, str(E)))


def load_concept(word):
    path = f"./data/concepts/{word}.json"
    with open(path, "r") as f:
        return json.load(f)
