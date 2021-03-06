import torch.nn as nn
from datasets import load_metric
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    logging,
)

from cl.data import preprocess_for_meta_cf, preprocess_for_ssl
from cl.models import (
    FastBertForSequenceClassification,
    FastGPT2ForSequenceClassification,
)
from cl.utils import DataPreprocessing

logging.set_verbosity_error()

FastModel = {
    "bert-base-uncased": FastBertForSequenceClassification,
    "distilgpt2": FastGPT2ForSequenceClassification,
}


class DualNet:
    def __init__(self, args):
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast_tokenizer=args.use_fast_tokenizer,
        )
        self.slow_learner = SlowLearner(args, tokenizer=self.tokenizer)
        self.slow_learner.lm.resize_token_embeddings(len(self.tokenizer))

        self.fast_learner = FastLearner(
            args, tokenizer=self.tokenizer, slow_learner=self.slow_learner
        )


class SlowLearner(nn.Module):
    def __init__(self, args, tokenizer):
        super(SlowLearner, self).__init__()
        self.args = args

        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.lm = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, config=self.config
        )

        self.tokenizer = tokenizer
        self.preprocessor = DataPreprocessing(
            block_size=min(tokenizer.model_max_length, args.max_length),
            metric=load_metric("accuracy"),
        )
        if "gpt" in args.model_name_or_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

    def ssl_semantic(self):
        raise NotImplementedError

    def ssl_causal_lm(self, data, epoch):
        train_dataset, eval_dataset = preprocess_for_ssl(
            data, self.preprocessor, self.tokenizer, self.args
        )

        training_args = TrainingArguments(
            output_dir=f"{self.args.output_dir}/ssl_causal_lm",
            num_train_epochs=self.args.lm_epochs,
            per_device_eval_batch_size=self.args.lm_eval_batch_size,
            per_device_train_batch_size=self.args.lm_train_batch_size,
            lr_scheduler_type="cosine",
            learning_rate=self.args.lm_lr,
        )

        trainer = Trainer(
            model=self.lm,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.lm_collator,
            compute_metrics=self.preprocessor.compute_metrics,
            preprocess_logits_for_metrics=self.preprocessor.preprocess_logits_for_metrics,
        )
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        print(metrics, epoch)

    def forward(self, input_id, mask):
        if "bert" in self.args.model_name_or_path:
            lm = self.lm.bert
        elif "gpt" in self.args.model_name_or_path:
            lm = self.lm.transformer
        _, x, h = lm(
            input_ids=input_id,
            attention_mask=mask,
            return_dict=False,
            output_hidden_states=True,
        )
        return x, h[1:]


class FastLearner(nn.Module):
    def __init__(self, args, tokenizer, slow_learner):
        super(FastLearner, self).__init__()
        self.args = args
        self.slow_learner = slow_learner

        self.tokenizer = tokenizer

        num_labels = args.n_ways if args.task_aware else args.n_classes

        self.cf = FastModel[args.model_name_or_path].from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
        )
        if "gpt" in self.args.model_name_or_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token_id
            self.cf.config.pad_token_id = self.cf.config.eos_token_id

    def forward(self, input_ids, attention_mask, labels, *args):
        _, h = self.slow_learner(input_ids, attention_mask)
        return self.cf(
            h,
            self.class_weights,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            labels=labels,
            *args,
        )

    def meta_cf(self, data, epoch, token_type_ids=None):
        train_dataset, eval_dataset, _ = preprocess_for_meta_cf(
            data, self.tokenizer, self.args
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=f"{self.args.output_dir}/meta_cf",
            learning_rate=self.args.meta_lr,
            per_device_train_batch_size=self.args.meta_train_batch_size,
            per_device_eval_batch_size=self.args.meta_eval_batch_size,
            num_train_epochs=self.args.meta_epochs,
            weight_decay=self.args.meta_weight_decay,
        )

        self.class_weights = (
            (data.offset, self.args.n_ways) if self.args.task_aware else None
        )
        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
