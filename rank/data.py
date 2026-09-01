"""Length-aware batch sampling for reranker training."""

import random

import numpy as np
import torch


class SmartBatchingSampler(torch.utils.data.Sampler):
    """Group similarly sized examples to reduce padding during training."""

    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

        if "length" not in data_source.column_names:
            indices = list(range(len(data_source)))
            random.shuffle(indices)
            self.batches = [
                indices[i:i + batch_size]
                for i in range(0, len(indices), batch_size)
            ]
            return

        lengths = np.asarray(data_source["length"])
        indices = np.argsort(lengths)
        self.batches = [
            indices[i:i + batch_size]
            for i in range(0, len(indices), batch_size)
        ]
        random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield from batch

    def __len__(self):
        return len(self.data_source)
