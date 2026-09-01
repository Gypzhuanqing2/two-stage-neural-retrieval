"""Ranking metrics and NanoBEIR data preparation for reranker evaluation."""

import csv
import os

import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score, ndcg_score
from tqdm.auto import tqdm


class CrossEncoderRerankingEvaluator:
    """Evaluate a Cross-Encoder on query-document ranking samples."""

    def __init__(
        self,
        samples,
        at_k=10,
        name="",
        batch_size=4,
        show_progress_bar=False,
        write_csv=True,
        mrr_at_k=None,
    ):
        self.samples = samples
        self.at_k = at_k if mrr_at_k is None else mrr_at_k
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv
        self.csv_file = f"CrossEncoderRerankingEvaluator_{name}_results.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "MAP",
            f"MRR@{self.at_k}",
            f"NDCG@{self.at_k}",
        ]

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        if epoch != -1:
            status = (
                f"after epoch {epoch}"
                if steps == -1
                else f"after epoch {epoch}, step {steps}"
            )
        else:
            status = "current checkpoint"

        print(f"Evaluating {status} on {self.name}")

        mrr_scores = []
        ndcg_scores = []
        ap_scores = []
        iterator = tqdm(
            self.samples,
            desc="Evaluating samples",
            disable=not self.show_progress_bar,
        )

        for instance in iterator:
            query = instance["query"]
            positive = instance["positive"]
            positive = [positive] if isinstance(positive, str) else positive
            documents = instance.get("documents", [])
            negative = instance.get("negative", [])

            if documents:
                documents_to_score = documents
                relevance = [
                    int(document in positive)
                    for document in documents_to_score
                ]
            else:
                documents_to_score = positive + negative
                relevance = [1] * len(positive) + [0] * len(negative)

            if not documents_to_score:
                continue

            pairs = [[query, document] for document in documents_to_score]
            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            mrr, ndcg, ap = self.compute_metrics(relevance, scores)
            mrr_scores.append(mrr)
            ndcg_scores.append(ndcg)
            ap_scores.append(ap)

        mean_mrr = self._mean_or_zero(mrr_scores)
        mean_ndcg = self._mean_or_zero(ndcg_scores)
        mean_ap = self._mean_or_zero(ap_scores)
        metrics = {
            "map": mean_ap,
            f"mrr@{self.at_k}": mean_mrr,
            f"ndcg@{self.at_k}": mean_ndcg,
        }

        print(f"MAP: {mean_ap * 100:.2f}")
        print(f"MRR@{self.at_k}: {mean_mrr * 100:.2f}")
        print(f"NDCG@{self.at_k}: {mean_ndcg * 100:.2f}")

        if output_path is not None and self.write_csv:
            self._write_csv(
                output_path,
                epoch,
                steps,
                mean_ap,
                mean_mrr,
                mean_ndcg,
            )

        return metrics

    @staticmethod
    def _mean_or_zero(values):
        return float(np.mean(values)) if values else 0.0

    def _write_csv(self, output_path, epoch, steps, mean_ap, mean_mrr, mean_ndcg):
        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(output_path, self.csv_file)
        exists = os.path.isfile(csv_path)
        with open(
            csv_path,
            mode="a" if exists else "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(self.csv_headers)
            writer.writerow([epoch, steps, mean_ap, mean_mrr, mean_ndcg])

    def compute_metrics(self, y_true, y_pred):
        """Return MRR, NDCG, and average precision for one query."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if np.sum(y_true) == 0:
            return 0.0, 0.0, 0.0

        ranking = np.argsort(y_pred)[::-1]
        mrr = 0.0
        for rank, index in enumerate(ranking[:self.at_k]):
            if y_true[index] > 0:
                mrr = 1.0 / (rank + 1)
                break

        ndcg = (
            ndcg_score([y_true], [y_pred], k=self.at_k)
            if len(y_true) > 1
            else 0.0
        )
        ap = average_precision_score(y_true, y_pred)
        return mrr, ndcg, ap


_NANOBEIR_NAMES = {
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "fiqa2018": "FiQA2018",
    "scidocs": "SCIDOCS",
    "arguana": "ArguAna",
    "touche2020": "Touche2020",
    "climatefever": "ClimateFEVER",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "quoraretrieval": "QuoraRetrieval",
    "scifact": "SciFact",
}


class CrossEncoderNanoBEIREvaluator:
    """Load NanoBEIR subsets and evaluate them with a reranker evaluator."""

    def __init__(
        self,
        dataset_names,
        dataset_id="sentence-transformers/NanoBEIR-en",
        rerank_k=100,
        at_k=10,
        batch_size=4,
    ):
        self.dataset_names = dataset_names
        self.dataset_id = dataset_id
        self.rerank_k = rerank_k
        self.at_k = at_k
        self.batch_size = batch_size
        self.evaluators = [self._load_dataset(name) for name in dataset_names]

    def _load_dataset(self, dataset_name):
        print(f"Loading NanoBEIR dataset: {dataset_name}")
        human_readable = _NANOBEIR_NAMES.get(
            dataset_name.lower(),
            dataset_name,
        )
        split_name = f"Nano{human_readable}"

        corpus = load_dataset(self.dataset_id, "corpus", split=split_name)
        queries = load_dataset(self.dataset_id, "queries", split=split_name)
        qrels = load_dataset(self.dataset_id, "qrels", split=split_name)
        bm25 = load_dataset(self.dataset_id, "bm25", split=split_name)

        corpus_mapping = {item["_id"]: item["text"] for item in corpus}
        query_mapping = {item["_id"]: item["text"] for item in queries}
        qrels_mapping = {}
        for item in qrels:
            qid = item["query-id"]
            cid = item["corpus-id"]
            qrels_mapping.setdefault(qid, set())
            if isinstance(cid, list):
                qrels_mapping[qid].update(cid)
            else:
                qrels_mapping[qid].add(cid)

        samples = []
        for item in bm25:
            qid = item["query-id"]
            if qid not in query_mapping:
                continue

            positive_ids = qrels_mapping.get(qid, set())
            positive_texts = [
                corpus_mapping[pid]
                for pid in positive_ids
                if pid in corpus_mapping
            ]
            candidate_ids = item["corpus-ids"][:self.rerank_k]
            candidate_texts = [
                corpus_mapping[cid]
                for cid in candidate_ids
                if cid in corpus_mapping
            ]
            samples.append({
                "query": query_mapping[qid],
                "positive": positive_texts,
                "documents": candidate_texts,
            })

        return CrossEncoderRerankingEvaluator(
            samples=samples,
            name=f"Nano{human_readable}",
            at_k=self.at_k,
            batch_size=self.batch_size,
        )

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        results = {}
        all_mrr = []
        all_ndcg = []
        for evaluator in self.evaluators:
            metrics = evaluator(model, output_path, epoch, steps)
            for key, value in metrics.items():
                results[f"{evaluator.name}_{key}"] = value

            mrr_key = f"mrr@{self.at_k}"
            ndcg_key = f"ndcg@{self.at_k}"
            if mrr_key in metrics:
                all_mrr.append(metrics[mrr_key])
            if ndcg_key in metrics:
                all_ndcg.append(metrics[ndcg_key])

        if all_mrr:
            results[f"NanoBEIR_R{self.rerank_k}_mean_mrr@{self.at_k}"] = np.mean(all_mrr)
        if all_ndcg:
            results[f"NanoBEIR_R{self.rerank_k}_mean_ndcg@{self.at_k}"] = np.mean(all_ndcg)
        return results
