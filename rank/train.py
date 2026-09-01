"""Train the Cross-Encoder reranker on MS MARCO."""

import argparse
import os

import torch
from datasets import load_dataset

if __package__:
    from .evaluation import CrossEncoderNanoBEIREvaluator
    from .model import CrossEncoder
    from .trainer import BCETrainer
else:
    from evaluation import CrossEncoderNanoBEIREvaluator
    from model import CrossEncoder
    from trainer import BCETrainer


def parse_args():
    """Parse the public training options and provide reproducible defaults."""
    parser = argparse.ArgumentParser(
        description="Train a Cross-Encoder reranker on MS MARCO."
    )
    parser.add_argument(
        "--model-name-or-path",
        default="answerdotai/ModernBERT-base",
        help="Hugging Face model id or local base-model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/reranker-msmarco-v1.1-ModernBERT-base-bce",
    )
    parser.add_argument("--epochs", dest="num_train_epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-steps", type=int, default=4000)
    parser.add_argument("--save-steps", type=int, default=4000)
    parser.add_argument("--logging-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--no-group-by-length",
        action="store_false",
        dest="group_by_length",
        help="Disable length-aware batching.",
    )
    parser.set_defaults(
        load_best_model_at_end=True,
        metric_for_best_model="NanoBEIR_R100_mean_ndcg@10",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        logging_first_step=True,
        group_by_length=True,
    )
    return parser.parse_args()


def bce_mapper(batch):
    """Flatten nested MS MARCO passages into query/passage/label rows."""
    queries = []
    passages = []
    labels = []
    for query, passage_info in zip(batch["query"], batch["passages"]):
        for index, label in enumerate(passage_info["is_selected"]):
            queries.append(query)
            passages.append(passage_info["passage_text"][index])
            labels.append(label)
    return {"query": queries, "passage": passages, "label": labels}


def add_length(batch):
    """Add an approximate text length used by the smart batching sampler."""
    return {
        "length": [
            len(query) + len(passage)
            for query, passage in zip(batch["query"], batch["passage"])
        ]
    }


def prepare_datasets(group_by_length=True):
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")
    dataset = dataset.train_test_split(test_size=1000)
    for split in ("train", "test"):
        dataset[split] = dataset[split].map(
            bce_mapper,
            batched=True,
            remove_columns=dataset[split].column_names,
        )

    train_dataset = dataset["train"]
    if group_by_length:
        train_dataset = train_dataset.map(add_length, batched=True)
    return train_dataset, dataset["test"]


def make_collate_fn(tokenizer, max_length):
    def collate(batch):
        queries = [item["query"] for item in batch]
        passages = [item["passage"] for item in batch]
        labels = [item["label"] for item in batch]
        features = tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        features["labels"] = torch.tensor(labels, dtype=torch.float)
        return features

    return collate


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    train_dataset, eval_dataset = prepare_datasets(args.group_by_length)
    torch.manual_seed(args.seed)
    model = CrossEncoder(args.model_name_or_path)
    evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"],
        batch_size=4,
    )
    trainer = BCETrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_collate_fn(model.tokenizer, model.max_length),
        evaluator=evaluator,
    )

    trainer.train()
    evaluator(model)
    model.save_pretrained(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
