"""Run dense retrieval followed by Cross-Encoder reranking."""

import argparse
import gzip
import json
import os
import pickle

import faiss
import numpy as np
import torch
from sentence_transformers import util
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__:
    from .model import CrossEncoder
    from .recall_model import BiEncoderBatchNeg
else:
    from model import CrossEncoder
    from recall_model import BiEncoderBatchNeg


TOP_K_RETRIEVAL = 8
MAX_SEQ_LENGTH_RECALL = 256
INDEX_FILENAME = "wikipedia_corpus.faiss"
PASSAGES_FILENAME = "wikipedia_passages.pkl"
WIKIPEDIA_FILENAME = "simplewiki-2020-11-01.jsonl.gz"


def encode_corpus(
    model,
    tokenizer,
    passages,
    device,
    max_seq_length=MAX_SEQ_LENGTH_RECALL,
    batch_size=64,
):
    """Encode all passages as a CPU tensor of normalized embeddings."""
    embeddings = []
    for start in tqdm(range(0, len(passages), batch_size), desc="Encoding corpus"):
        batch = passages[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            batch_embeddings = model.get_pooled_embedding(
                inputs["input_ids"],
                inputs["attention_mask"],
            )
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)


def load_or_build_index(
    model,
    tokenizer,
    device,
    index_path,
    passages_path,
    wikipedia_path,
    max_seq_length=MAX_SEQ_LENGTH_RECALL,
):
    """Load the paired FAISS index/passages cache or build it from SimpleWiki."""
    if os.path.exists(index_path) and os.path.exists(passages_path):
        # FAISS returns row ids, so this cache must stay paired with passages.
        print(f"Loading existing FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        print(f"Loading existing passages from {passages_path}...")
        with open(passages_path, "rb") as file:
            passages = pickle.load(file)
        print(f"Total Passages Loaded: {len(passages)}")
        return index, passages

    print("Index not found. Building new data and FAISS index...")
    if not os.path.exists(wikipedia_path):
        util.http_get(
            "https://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz",
            wikipedia_path,
        )

    passages = []
    print("Reading Wikipedia passages")
    with gzip.open(wikipedia_path, "rt", encoding="utf8") as file:
        for line in file:
            data = json.loads(line)
            passages.append(data["paragraphs"][0])
    print(f"Total passages: {len(passages)}")

    with open(passages_path, "wb") as file:
        pickle.dump(passages, file)

    corpus_embeddings = encode_corpus(
        model,
        tokenizer,
        passages,
        device=device,
        max_seq_length=max_seq_length,
    )
    embeddings = corpus_embeddings.numpy().astype(np.float32)
    index = faiss.IndexHNSWFlat(
        embeddings.shape[1],
        256,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.hnsw.efConstruction = 100
    index.add(embeddings)
    print(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)
    return index, passages


def load_pipeline(
    recall_model_path,
    reranker_base_model,
    reranker_checkpoint,
    cache_dir,
    device_name="auto",
):
    """Load trained models and the cached index required by ``search``."""
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    recall_tokenizer = AutoTokenizer.from_pretrained(recall_model_path)
    bi_encoder = BiEncoderBatchNeg(
        recall_model_path,
        max_seq_length=MAX_SEQ_LENGTH_RECALL,
    ).to(device)
    bi_encoder.eval()
    cross_encoder = CrossEncoder(
        base_model_name_or_path=reranker_base_model,
        checkpoint_path=reranker_checkpoint,
        max_length=512,
        device=str(device),
    )
    cross_encoder.eval()
    index, passages = load_or_build_index(
        bi_encoder,
        recall_tokenizer,
        device,
        index_path=os.path.join(cache_dir, INDEX_FILENAME),
        passages_path=os.path.join(cache_dir, PASSAGES_FILENAME),
        wikipedia_path=os.path.join(cache_dir, WIKIPEDIA_FILENAME),
    )
    return recall_tokenizer, bi_encoder, cross_encoder, index, passages, device


def search(
    query,
    recall_tokenizer=None,
    bi_encoder=None,
    cross_encoder=None,
    index=None,
    passages=None,
    device=None,
    top_k=TOP_K_RETRIEVAL,
    max_seq_length=MAX_SEQ_LENGTH_RECALL,
):
    """Return candidates ranked by the Cross-Encoder score."""
    components = (
        recall_tokenizer,
        bi_encoder,
        cross_encoder,
        index,
        passages,
        device,
    )
    if any(component is None for component in components):
        raise ValueError("Provide all model, index, passage, and device components.")

    print(f"\nQuery: {query}")
    query_inputs = recall_tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    query_inputs = {
        key: value.to(device)
        for key, value in query_inputs.items()
    }
    with torch.no_grad():
        query_embedding = bi_encoder.get_pooled_embedding(
            query_inputs["input_ids"],
            query_inputs["attention_mask"],
        )

    index.hnsw.efSearch = max(64, top_k)
    scores, indices = index.search(
        query_embedding.cpu().numpy().astype(np.float32),
        top_k,
    )
    hits = [
        {
            "corpus_id": int(index_id),
            "score": float(score),
            "text": passages[index_id],
        }
        for score, index_id in zip(scores[0], indices[0])
    ]

    print("\n-------------------------")
    print(f"Top-3 retrieval results ({top_k} candidates)")
    for hit in hits[:3]:
        print(f"{hit['score']:.3f}  {hit['text'].replace(chr(10), ' ')}")

    pairs = [[query, hit["text"]] for hit in hits]
    cross_scores = cross_encoder.predict(pairs, batch_size=32)
    for hit, score in zip(hits, cross_scores):
        hit["cross-score"] = float(score)
    reranked = sorted(hits, key=lambda hit: hit["cross-score"], reverse=True)

    print("\n-------------------------")
    print("Top-3 reranked results")
    for hit in reranked[:3]:
        print(f"{hit['cross-score']:.3f}  {hit['text'].replace(chr(10), ' ')}")
    return reranked


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Bi-Encoder retrieval followed by Cross-Encoder reranking."
    )
    parser.add_argument(
        "--query",
        default="What is the capital of the United States?",
    )
    parser.add_argument(
        "--recall-model",
        required=True,
        help="Fine-tuned Bi-Encoder directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--reranker-base-model",
        default="answerdotai/ModernBERT-base",
    )
    parser.add_argument(
        "--reranker-checkpoint",
        required=True,
        help="Directory containing the fine-tuned Cross-Encoder weights.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
    )
    parser.add_argument("--top-k", type=int, default=TOP_K_RETRIEVAL)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    components = load_pipeline(
        recall_model_path=args.recall_model,
        reranker_base_model=args.reranker_base_model,
        reranker_checkpoint=args.reranker_checkpoint,
        cache_dir=args.cache_dir,
        device_name=args.device,
    )
    search(args.query, *components, top_k=args.top_k)


if __name__ == "__main__":
    main()
