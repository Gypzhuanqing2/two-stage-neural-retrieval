"""Bi-Encoder used by the search-time dense retriever."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class BiEncoderBatchNeg(nn.Module):
    """Match the training encoder so queries share the index embedding space."""

    def __init__(self, model_name_or_path, max_seq_length=300, scale=20.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
        )
        self.max_seq_length = max_seq_length
        self.scale = scale
        self.loss_fct = nn.CrossEntropyLoss()

    def get_pooled_embedding(self, input_ids, attention_mask):
        """Return masked-mean, L2-normalized sentence embeddings."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        return F.normalize(sentence_embeddings, p=2, dim=1)

    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        pos_input_ids,
        pos_attention_mask,
        neg_input_ids,
        neg_attention_mask,
    ):
        query_embeddings = self.get_pooled_embedding(
            query_input_ids,
            query_attention_mask,
        )
        pos_embeddings = self.get_pooled_embedding(
            pos_input_ids,
            pos_attention_mask,
        )
        neg_embeddings = self.get_pooled_embedding(
            neg_input_ids,
            neg_attention_mask,
        )

        candidates = torch.cat([pos_embeddings, neg_embeddings], dim=0)
        # Query i is matched with candidate i; the rest are in-batch negatives.
        scores = torch.matmul(query_embeddings, candidates.transpose(0, 1))
        scores = scores * self.scale
        labels = torch.arange(
            len(query_embeddings),
            dtype=torch.long,
            device=scores.device,
        )
        return self.loss_fct(scores, labels)
