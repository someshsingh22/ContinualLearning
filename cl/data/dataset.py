from datasets import load_dataset


class MetaLoader(object):
    """
    Generator of datasets for the given dataset.
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.dataset = load_dataset(
            args.data_args.dataset_name, args.data_args.dataset_config_name
        )
        self.tokenizer = tokenizer
        self.task_datasets = []
        for label in range(0, args.num_classes, args.num_classes_per_loader):
            start, end = label, label + args.num_classes_per_loader
            task_datasets = self.dataset.filter(
                lambda x: x["intent"] in range(start, end)
            )
            self.task_datasets.append(task_datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]

    def __len__(self):
        return len(self.datasets)


def preprocess_for_ssl(task_datasets, Preprocessor, tokenizer, args):
    tokenized_task_datasets = task_datasets.map(
        lambda examples: tokenizer(examples[args.text_column_name]),
        batched=True,
        desc="Running tokenizer on dataset",
    )

    lm_datasets = tokenized_task_datasets.map(
        Preprocessor.group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {Preprocessor.block_size}",
    )
    train_dataset, eval_dataset = (
        lm_datasets["train"],
        lm_datasets["validation"],
    )
    return train_dataset, eval_dataset


def preprocess_for_meta_cf(task_datasets, tokenizer, args):
    meta_datasets = task_datasets.map(
        lambda examples: tokenizer(examples[args.text_column_name], truncation=True),
        batched=True,
        desc="Running tokenizer on dataset",
    )

    train_dataset, eval_dataset, test_dataset = (
        meta_datasets["train"],
        meta_datasets["validation"],
        meta_datasets["test"],
    )

    return train_dataset, eval_dataset, test_dataset
