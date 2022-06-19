# Imports
import argparse
import logging

import datasets
from datasets import load_metric

from cl.data import MetaTaskLoader
from cl.models import DualNet
from cl.utils import DataPreprocessing

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.FileHandler("./logs/main.log")],
)
logger = logging.getLogger("main")

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./results")
parser.add_argument("--model_name_or_path", type=str, default="distilgpt2")
parser.add_argument("--dataset_name", type=str, default="clinc_oos")
parser.add_argument("--dataset_config_name", type=str, default="plus")
parser.add_argument("--n_classes", type=int, default=150)
parser.add_argument("--n_ways", type=int, default=5)
parser.add_argument("--text_column_name", type=str, default="text")
parser.add_argument("--label_column_name", type=str, default="intent")
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--meta_train_batch_size", type=int, default=32)
parser.add_argument("--meta_lr", type=float, default=1e-3)
parser.add_argument("--meta_epochs", type=int, default=1)
parser.add_argument("--meta_eval_batch_size", type=int, default=32)
parser.add_argument("--meta_weight_decay", type=float, default=0.1)
parser.add_argument("--lm_train_batch_size", type=int, default=32)
parser.add_argument("--lm_lr", type=float, default=1e-3)
parser.add_argument("--lm_epochs", type=int, default=1)
parser.add_argument("--lm_eval_batch_size", type=int, default=32)
parser.add_argument("--use_fast_tokenizer", action="store_true")
parser.add_argument("--task_aware", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


# Main
if __name__ == "__main__":
    model = DualNet(args)
    tokenizer = model.tokenizer
    MTL = MetaTaskLoader(args, tokenizer=tokenizer)

    # Init variables

    for task_id, task in enumerate(MTL):

        model.slow_learner.ssl_causal_lm(task, epoch=task_id)
        # model.slow_learner.ssl_semantic(task, epoch=task_id)
        model.fast_learner.meta_cf(task, epoch=task_id)

        ## Train
        ### SSL

        ### DL

        ## Test
        ### DL
        pass
