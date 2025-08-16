from datasets import load_dataset

# Streaming mode
dataset_stream = load_dataset("sentence-transformers/parallel-sentences-europarl", "en-de", split="train", streaming=True)

# Take first 1000 examples
subset = []
for i, example in enumerate(dataset_stream):
    if i >= 1000:
        break
    subset.append(example)

print(len(subset))
print(subset[0])
