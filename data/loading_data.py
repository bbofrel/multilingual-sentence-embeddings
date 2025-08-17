from datasets import load_dataset

def load_dataset_subset():
    dataset_stream = load_dataset("sentence-transformers/parallel-sentences-europarl", "en-de", split="train", streaming=True)

    subset = []
    for i, example in enumerate(dataset_stream):
        if i >= 1000:
            break
        subset.append(example)
    return subset
