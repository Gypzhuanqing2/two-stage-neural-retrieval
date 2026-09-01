"""Cross-Encoder model used for second-stage reranking."""

import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class CrossEncoder(nn.Module):
    """Score query-document pairs with a Transformer and classification head."""

    def __init__(self, model_name_or_path, max_length=256, device=None):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.config = self.base_model.config
        self.max_length = max_length

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.dense)
        self._init_weights(self.classifier)

        weights_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            print(f"Loading custom weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

        self.device = torch.device(
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def _init_weights(self, module):
        if not isinstance(module, nn.Linear):
            return
        module.weight.data.normal_(
            mean=0.0,
            std=getattr(self.config, "initializer_range", 0.02),
        )
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        if outputs.last_hidden_state is not None:
            pooled = outputs.last_hidden_state[:, 0, :]
        elif getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            raise ValueError(
                "Model outputs contain neither last_hidden_state nor pooler_output."
            )

        hidden = self.dropout(pooled)
        hidden = self.activation(self.dense(hidden))
        hidden = self.dropout(hidden)
        return self.classifier(hidden)

    def predict(
        self,
        sentences,
        batch_size=32,
        activation_fct=None,
        show_progress_bar=False,
    ):
        """Score one pair or a list of ``[query, document]`` pairs."""
        if not sentences:
            return torch.tensor([]).numpy()

        self.eval()
        input_was_singular = isinstance(sentences[0], str)
        if input_was_singular:
            sentences = [sentences]

        activation_fct = activation_fct or nn.Sigmoid()
        all_scores = []
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                batch = sentences[start_index:start_index + batch_size]
                queries = [pair[0] for pair in batch]
                documents = [pair[1] for pair in batch]
                features = self.tokenizer(
                    queries,
                    documents,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                features = {
                    key: value.to(self.device)
                    for key, value in features.items()
                }
                scores = activation_fct(self(**features)).view(-1)
                all_scores.append(scores.cpu())

        scores = torch.cat(all_scores).numpy()
        return scores[0] if input_was_singular else scores

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(
            self.state_dict(),
            os.path.join(save_directory, "pytorch_model.bin"),
        )
        self.config.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
