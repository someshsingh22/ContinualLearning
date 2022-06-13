#Imports
import learn2learn as l2l
import random
import argparse
#Global Variables


#Functions


#Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="./data/")
args = parser.parse_args()

#Main
if __name__ == "__main__":
    #Import AGNews Dataset
    train_dataset=Dataset('data/data_full.json','oos_train')
test_dataset=Dataset('data/data_full.json','oos_test')
train_data, test_data = l2l.data.MetaDataset(train_dataset), l2l.data.MetaDataset(
    test_dataset
)


transforms = [
    ContinousNWays(train_data,n_ways=5),
    l2l.data.transforms.LoadData(train_data),
]
train_taskset = TaskDataset(
    train_data, transforms, num_tasks=n_class // 5
)
test_taskset = TaskDataset(
    test_data, transforms, num_tasks=n_class // 5
)

# Generator function for a range
def task_gen(dataset, class_label='intent', start=0, end=150, n_ways=5):
    for i in range(start, end, n_ways):
        filter = lambda ex: ex[class_label]>= start and ex[class_label]<end
        yield dataset.filter(filter)
    
    #Implement Model
    model_args, data_args, training_args = get_args(output_dir='./', dataset_name='clinc_oos')

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,  use_fast_tokenizer=model_args.use_fast_tokenizer)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    set_seed(42)


    text_column_name = "text"
    max_seq_length = 64
    padding = "max_length"
    raw_datasets = load_dataset("clinc_oos", "plus")
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenizer(examples[text_column_name]),
        batched=True,
        desc="Running tokenizer on dataset",
    )
    Preprocessor = DataPreprocessing(block_size=min(tokenizer.model_max_length, 1024), metric = load_metric("accuracy"))
    
    lm_datasets = tokenized_datasets.map(
        Preprocessor.group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {Preprocessor.block_size}",
    )
    train_dataset, eval_dataset, test_dataset = lm_datasets["train"], lm_datasets["validation"], lm_datasets["test"]

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

    #Implemnet Training Loop
    
    pass