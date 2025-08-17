from data.loading_data import load_dataset_subset

def preprocessing(example):
    example = example.strip()
    example = " ".join(example.split())
    return example


def dataset_preprocessing():
    dataset = load_dataset_subset()
    # let's say len(sentence) should be bigger than 2 to be long enough
    processed_dataset = [(pair['english'], pair['non_english']) for pair in dataset if len(pair['english'])>2 and len(pair['non_english'])>2]
    processed_dataset = [(preprocessing(s1), preprocessing(s2)) for s1, s2 in processed_dataset]
    return processed_dataset



