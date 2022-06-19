# Save request json output
import json
import logging

import requests


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
