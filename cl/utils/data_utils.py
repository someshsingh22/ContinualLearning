from itertools import chain


class DataPreprocessing:
    def __init__(self, block_size, metric):
        self.block_size = block_size
        self.metric = metric

    def group_texts(self, examples):
        def concatenator(x):
            return list(chain(*x)) if isinstance(x[0], list) else x

        concatenated_examples = {k: concatenator(examples[k]) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return self.metric.compute(predictions=preds, references=labels)
