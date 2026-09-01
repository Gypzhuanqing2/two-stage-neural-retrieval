"""Fine-tune the first-stage Bi-Encoder with MS MARCO hard negatives."""

import argparse
import gzip
import json
import logging
import os
import pickle
import random
import tarfile
import time

import requests
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

if __package__:
    from .recall_model import BiEncoderBatchNeg
else:
    from recall_model import BiEncoderBatchNeg


COLLECTION_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
)
QUERIES_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"
)
HARD_NEGATIVES_BASE_URL = (
    "https://huggingface.co/datasets/sentence-transformers/"
    "msmarco-hard-negatives/resolve/main"
)


def http_get(url, path):
    """Download ``url`` to ``path`` when the file is not already present."""
    if os.path.exists(path):
        return
    logging.info("Downloading %s to %s", url, path)
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        response.raise_for_status()
    with open(path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Bi-Encoder on MS MARCO hard negatives."
    )
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=300, type=int)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument(
        "--negs_to_use",
        default=None,
        help="Comma-separated hard-negative systems; use all systems by default.",
    )
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_negs_per_system", default=5, type=int)
    parser.add_argument("--ce_score_margin", default=3.0, type=float)
    return parser.parse_args()


def ensure_archive_file(data_folder, filename, archive_name, url):
    """Download and extract a tar.gz archive when its target file is absent."""
    target_path = os.path.join(data_folder, filename)
    if os.path.exists(target_path):
        return target_path

    archive_path = os.path.join(data_folder, archive_name)
    if not os.path.exists(archive_path):
        http_get(url, archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(path=data_folder)
    return target_path


def load_corpus(data_folder):
    collection_path = ensure_archive_file(
        data_folder,
        "collection.tsv",
        "collection.tar.gz",
        COLLECTION_URL,
    )
    print("Reading corpus: collection.tsv")
    corpus = {}
    with open(collection_path, encoding="utf8") as file:
        for line in file:
            pid, passage = line.rstrip("\n").split("\t", maxsplit=1)
            corpus[int(pid)] = passage
    return corpus


def load_queries(data_folder):
    queries_path = ensure_archive_file(
        data_folder,
        "queries.train.tsv",
        "queries.tar.gz",
        QUERIES_URL,
    )
    print("Reading queries: queries.train.tsv")
    queries = {}
    with open(queries_path, encoding="utf8") as file:
        for line in file:
            qid, query = line.rstrip("\n").split("\t", maxsplit=1)
            queries[int(qid)] = query
    return queries


def build_train_queries(data_folder, queries, args):
    scores_path = os.path.join(
        data_folder,
        "cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz",
    )
    http_get(
        f"{HARD_NEGATIVES_BASE_URL}/"
        "cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz",
        scores_path,
    )
    print("Loading Cross-Encoder scores")
    with gzip.open(scores_path, "rb") as file:
        ce_scores = pickle.load(file)

    hard_negatives_path = os.path.join(
        data_folder,
        "msmarco-hard-negatives.jsonl.gz",
    )
    http_get(
        f"{HARD_NEGATIVES_BASE_URL}/msmarco-hard-negatives.jsonl.gz",
        hard_negatives_path,
    )
    print("Reading hard negatives")

    systems = args.negs_to_use.split(",") if args.negs_to_use else None
    train_queries = {}
    with gzip.open(hard_negatives_path, "rt") as file:
        for line in tqdm(file):
            record = json.loads(line)
            qid = record["qid"]
            positive_ids = record["pos"]
            if not positive_ids:
                continue

            positive_score = min(ce_scores[qid][pid] for pid in positive_ids)
            threshold = positive_score - args.ce_score_margin
            # Drop negatives whose score is too close to a known positive.
            negative_ids = set()
            selected_systems = systems or list(record["neg"].keys())

            for system_name in selected_systems:
                if system_name not in record["neg"]:
                    continue
                added = 0
                for pid in record["neg"][system_name]:
                    if ce_scores[qid][pid] > threshold or pid in negative_ids:
                        continue
                    negative_ids.add(pid)
                    added += 1
                    if added >= args.num_negs_per_system:
                        break

            if negative_ids and qid in queries:
                train_queries[qid] = {
                    "qid": qid,
                    "query": queries[qid],
                    "pos": positive_ids,
                    "neg": list(negative_ids),
                }

    print(f"Training queries: {len(train_queries)}")
    return train_queries


class MSMARCODataset(Dataset):
    """Yield one rotating positive and hard-negative passage per query."""

    def __init__(self, train_queries, corpus):
        self.train_queries = list(train_queries.values())
        for query in self.train_queries:
            random.shuffle(query["neg"])
        self.corpus = corpus

    def __len__(self):
        return len(self.train_queries)

    def __getitem__(self, idx):
        query_data = self.train_queries[idx]
        pos_id = query_data["pos"].pop(0)
        query_data["pos"].append(pos_id)
        neg_id = query_data["neg"].pop(0)
        query_data["neg"].append(neg_id)
        return {
            "query": query_data["query"],
            "pos": self.corpus[pos_id],
            "neg": self.corpus[neg_id],
        }


def make_collate_fn(tokenizer, max_seq_length):
    def collate(batch):
        queries = [item["query"] for item in batch]
        positives = [item["pos"] for item in batch]
        negatives = [item["neg"] for item in batch]

        def tokenize(texts):
            return tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )

        query_tokens = tokenize(queries)
        positive_tokens = tokenize(positives)
        negative_tokens = tokenize(negatives)
        return {
            "query_input_ids": query_tokens["input_ids"],
            "query_attention_mask": query_tokens["attention_mask"],
            "pos_input_ids": positive_tokens["input_ids"],
            "pos_attention_mask": positive_tokens["attention_mask"],
            "neg_input_ids": negative_tokens["input_ids"],
            "neg_attention_mask": negative_tokens["attention_mask"],
        }

    return collate


def create_optimizer(model, learning_rate):
    no_decay = ("bias", "LayerNorm.weight")
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if any(term in name for term in no_decay):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "msmarco-data")
    os.makedirs(data_folder, exist_ok=True)

    corpus = load_corpus(data_folder)
    queries = load_queries(data_folder)
    train_queries = build_train_queries(data_folder, queries, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = MSMARCODataset(train_queries, corpus)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=make_collate_fn(tokenizer, args.max_seq_length),
        pin_memory=True,
    )

    model = BiEncoderBatchNeg(
        args.model_name,
        max_seq_length=args.max_seq_length,
    ).to(device)
    optimizer = create_optimizer(model, args.lr)
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    output_dir = os.path.join(
        script_dir,
        "output",
        "train_bi-encoder-mnrl-{}-{}".format(
            args.model_name.replace("/", "-"),
            time.strftime("%Y-%m-%d_%H-%M-%S"),
        ),
    )
    os.makedirs(output_dir, exist_ok=True)

    print("Starting training...")
    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 10 == 0:
                elapsed = time.time() - tic_train
                print(
                    f"Epoch: {epoch}, global step: {global_step}, "
                    f"batch: {step}, loss: {loss.item():.5f}, "
                    f"speed: {10 / elapsed:.2f} step/s"
                )
                tic_train = time.time()

        save_path = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        model.bert.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    print("Training finished.")


if __name__ == "__main__":
    train(parse_args())
