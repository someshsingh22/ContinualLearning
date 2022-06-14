"""
text_column_name = "text"
raw_datasets = load_dataset("clinc_oos", "plus")
tokenized_datasets = raw_datasets.map(
    lambda examples: tokenizer(examples[text_column_name]),
    batched=True,
    desc="Running tokenizer on dataset",
)
block_size = min(tokenizer.model_max_length, 1024)
"""
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


class MetaLoader(object):
    """
    Generator of datasets for the given dataset.
    """

    def __init__(self, args):
        self.args = args
        self.dataset = load_dataset(
            args.data_args.dataset_name, args.data_args.dataset_config_name
        )
        self.tasks = []
        for label in range(0, args.num_classes, args.num_classes_per_loader):
            start, end = label, label + args.num_classes_per_loader
            self.datasets.append(
                self.dataset.filter(lambda x: x["intent"] in range(start, end))
            )

    def __getitem__(self, idx):
        return self.datasets[idx]

    def __len__(self):
        return len(self.datasets)


def preprocess_for_ssl(tokenized_datasets, Preprocessor):
    lm_datasets = tokenized_datasets.map(
        Preprocessor.group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {Preprocessor.block_size}",
    )
    train_dataset, eval_dataset = (
        lm_datasets["train"],
        lm_datasets["validation"],
    )
    return train_dataset, eval_dataset
