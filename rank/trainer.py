"""Training loop for the reranker model."""

import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

if __package__:
    from .data import SmartBatchingSampler
else:
    from data import SmartBatchingSampler


class BCETrainer:
    """Run optimization, evaluation, and checkpoint saving for a Cross-Encoder."""

    def __init__(
        self,
        model,
        args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        evaluator=None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.evaluator = evaluator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.set_seed(self.args.seed)
        self.global_step = 0
        self.best_metric = (
            -float("inf") if args.metric_for_best_model else None
        )

    @staticmethod
    def set_seed(seed):
        """Seed Python, NumPy, and PyTorch random number generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_train_dataloader(self):
        if self.args.group_by_length and "length" in self.train_dataset.column_names:
            sampler = SmartBatchingSampler(
                self.train_dataset,
                self.args.train_batch_size,
            )
        else:
            sampler = RandomSampler(self.train_dataset)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=True,
        )

    def create_optimizer_and_scheduler(self, num_training_steps):
        no_decay = ("bias", "LayerNorm.weight")
        decay_params = []
        no_decay_params = []

        for name, parameter in self.model.named_parameters():
            if any(term in name for term in no_decay):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.args.weight_decay,
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * self.args.warmup_ratio),
            num_training_steps=num_training_steps,
        )

    @staticmethod
    def compute_loss(model, inputs):
        """Compute BCE loss from model logits and batch labels."""
        if "labels" not in inputs:
            raise ValueError("Labels missing in inputs for BCETrainer")
        labels = inputs.pop("labels")
        logits = model(**inputs)
        return nn.BCEWithLogitsLoss()(
            logits.view(-1),
            labels.float().view(-1),
        )

    def train(self):
        train_dataloader = self.get_train_dataloader()
        num_training_steps = len(train_dataloader) * self.args.num_train_epochs
        self.create_optimizer_and_scheduler(num_training_steps)

        print("***** Training *****")
        print(f"Training examples = {len(self.train_dataset)}")
        print(f"Epochs = {self.args.num_train_epochs}")
        print(f"Optimization steps = {num_training_steps}")

        progress_bar = tqdm(range(int(num_training_steps)))
        self.model.train()
        for epoch in range(int(self.args.num_train_epochs)):
            for _step, batch in enumerate(train_dataloader):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                loss = self.compute_loss(self.model, batch)
                loss.backward()

                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm,
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                progress_bar.update(1)

                if (
                    self.args.logging_steps > 0
                    and self.global_step % self.args.logging_steps == 0
                ):
                    print(f"Step {self.global_step}: Loss = {loss.item()}")

                if (
                    self.args.eval_steps > 0
                    and self.global_step % self.args.eval_steps == 0
                ):
                    self.evaluate()
                    self.save_model()

            if self.args.evaluation_strategy == "epoch":
                self.evaluate()
            if self.args.save_strategy == "epoch":
                self.save_model()

        progress_bar.close()

    def evaluate(self):
        if self.evaluator is None:
            return {}

        was_training = self.model.training
        try:
            metrics = self.evaluator(
                self.model,
                output_path=self.args.output_dir,
                epoch=self.global_step // len(self.get_train_dataloader()),
                steps=self.global_step,
            )
        finally:
            if was_training:
                self.model.train()

        metric_name = self.args.metric_for_best_model
        if metric_name and self.args.load_best_model_at_end and metric_name in metrics:
            score = metrics[metric_name]
            if score > self.best_metric:
                self.best_metric = score
                print(f"New best checkpoint: {metric_name} = {score}")
                self.save_model(is_best=True)

        return metrics

    def save_model(self, is_best=False):
        if is_best:
            output_dir = os.path.join(self.args.output_dir, "checkpoint-best")
        else:
            output_dir = os.path.join(
                self.args.output_dir,
                f"checkpoint-{self.global_step}",
            )

        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir)
