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
    Generator of dataloaders for the given dataset.
    """

    def __init__(self, dataset_name, args):
        self.args = args
        self.dataset = load_dataset(dataset_name)
        self.loaders = self.create_loaders(
            args.batch_size, args.shuffle, args.num_workers
        )

    def create_loaders(self, batch_size, shuffle, num_workers):
        """
        Create dataloaders for given huggingface dataset of given labels.
        """
        loaders = []
        for label in range(0, self.args.num_classes, self.args.num_classes_per_loader):
            pass
        return loaders

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)


def preprocess_for_ssl(tokenized_datasets, Preprocessor):
    lm_datasets = tokenized_datasets.map(
        Preprocessor.group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {Preprocessor.block_size}",
    )
    train_dataset, eval_dataset, test_dataset = (
        lm_datasets["train"],
        lm_datasets["validation"],
        lm_datasets["test"],
    )
    return train_dataset, eval_dataset, test_dataset
