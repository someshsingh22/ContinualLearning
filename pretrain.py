from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    default_data_collator,
    set_seed,
)

from cl.utils import DataPreprocessing, get_args

if __name__ == "__main__":

    model_args, data_args, training_args = get_args(
        output_dir="./", dataset_name="clinc_oos"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast_tokenizer=model_args.use_fast_tokenizer
    )
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, config=config
    )
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    set_seed(42)

    text_column_name = "text"
    raw_datasets = load_dataset("clinc_oos", "plus")
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenizer(examples[text_column_name]),
        batched=True,
        desc="Running tokenizer on dataset",
    )
    Preprocessor = DataPreprocessing(
        block_size=min(tokenizer.model_max_length, 1024), metric=load_metric("accuracy")
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=Preprocessor.compute_metrics,
        preprocess_logits_for_metrics=Preprocessor.preprocess_logits_for_metrics,
    )

    checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
