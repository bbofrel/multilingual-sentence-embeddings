from datasets import load_dataset
from itertools import islice


def load_dataset_subset():
    dataset_stream = load_dataset("sentence-transformers/parallel-sentences-europarl", "en-de", split="train",
                                  streaming=True)
    subset = []
    for i, example in enumerate(dataset_stream):
        if i >= 1000:
            break
        subset.append(example)
    return subset


def load_full_dataset():
    dataset_stream = load_dataset(
        "sentence-transformers/parallel-sentences-europarl",
        "en-de",
        split="train",
        streaming=True
    )
    dataset = list(islice(dataset_stream, 50000))
    return dataset
