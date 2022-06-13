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


def preprocess_for_ssl(tokenized_datasets, Preprocessor, args):
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
