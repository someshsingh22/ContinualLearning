from lib2to3.pgen2.tokenize import tokenize

import torch.nn as nn
from datasets import load_metric
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    default_data_collator,
)

from cl.data import preprocess_for_ssl
from cl.models import FastModel
from cl.utils import DataPreprocessing, get_args


class DualNet:
    def __init__(self, args):
        self.args = args
        args.model_args, args.data_args, args.training_args = get_args(
            output_dir=args.output_dir, dataset_name=args.dataset_name
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_args.model_name_or_path,
            use_fast_tokenizer=args.model_args.use_fast_tokenizer,
        )
        self.slow_learner = SlowLearner(args, tokenizer=self.tokenizer)
        self.slow_learner.resize_token_embeddings(len(self.tokenizer))

        self.fast_learner = FastLearner(args)


class SlowLearner(nn.Module):
    def __init__(self, args, tokenizer):
        super(SlowLearner, self).__init__()
        self.args = args

        self.config = AutoConfig.from_pretrained(args.model_args.model_name_or_path)
        self.lm = AutoModelForCausalLM.from_pretrained(
            args.model_args.model_name_or_path, config=self.config
        )
        self.tokenizer = tokenizer
        self.preprocessor = DataPreprocessing(
            block_size=min(tokenizer.model_max_length, args.max_length),
            metric=load_metric("accuracy"),
        )
        self.relu = nn.ReLU()

    def ssl_causal_lm(self, data, epoch):
        train_dataset, eval_dataset, test_dataset = preprocess_for_ssl(
            data,
            self.preprocessor,
        )
        trainer = Trainer(
            model=self.slow_learner,
            args=self.args.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            compute_metrics=self.preprocessor.compute_metrics,
            preprocess_logits_for_metrics=self.preprocessor.preprocess_logits_for_metrics,
        )
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        trainer.log_metrics(f"ssl_lm_epoch_{epoch}", metrics)
        trainer.save_metrics(f"ssl_lm_epoch_{epoch}", metrics)
        trainer.save_state()

    def forward(self, input_id, mask):
        _, h, x = self.lm(input_ids=input_id, attention_mask=mask, return_dict=False)
        x = self.relu(x)
        return x, h


class FastLearner(nn.Module):
    def __init__(self, args):
        super(FastLearner, self).__init__()
        self.args = args

        self.lm = FastModel[args.model_args.model_name_or_path].from_pretrained(
            args.model_args.model_name_or_path
        )
        self.linear = nn.Linear(args.h, args.out)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, x = self.lm(input_ids=input_id, attention_mask=mask, return_dict=False)
        x = self.linear(x)
        x = self.relu(x)

        return x
