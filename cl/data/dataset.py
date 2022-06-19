import datasets


class MetaTaskLoader(object):
    """
    Generator of datasets for the given dataset.
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.dataset = datasets.load_dataset(
            args.dataset_name, args.dataset_config_name
        )
        self.tokenizer = tokenizer
        self.task_datasets = []

        def unoffset(example):
            example[args.label_column_name] = [
                intent for intent in example[args.label_column_name]
            ]
            return example

        for label in range(0, args.n_classes, args.n_ways):
            start, end = label, label + args.n_ways
            task_datasets = self.dataset.filter(
                lambda x: x[args.label_column_name] in range(start, end)
            )
            task_datasets = task_datasets.map(unoffset, batched=True)
            task_datasets = task_datasets.shuffle()
            task_datasets.offset = start
            self.task_datasets.append(task_datasets)

    def __getitem__(self, idx):
        return self.task_datasets[idx]

    def __len__(self):
        return len(self.task_datasets)


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
    meta_datasets = meta_datasets.map(
        lambda examples: {"labels": examples[args.label_column_name]}, batched=True
    )

    train_dataset, eval_dataset, test_dataset = (
        meta_datasets["train"],
        meta_datasets["validation"],
        meta_datasets["test"],
    )

    return train_dataset, eval_dataset, test_dataset
