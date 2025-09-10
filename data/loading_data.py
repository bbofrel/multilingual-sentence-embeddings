from datasets import load_dataset
from itertools import islice


def load_dataset_subset(n_samples: int = 1000):
    dataset_stream = load_dataset("sentence-transformers/parallel-sentences-europarl", "en-de", split="train",
                                  streaming=True)
    return list(islice(dataset_stream, n_samples))


def load_full_dataset(limit: int = 50000):
    dataset_stream = load_dataset(
        "sentence-transformers/parallel-sentences-europarl",
        "en-de",
        split="train",
        streaming=True
    )
    return list(islice(dataset_stream, limit))
