from data.loading_data import load_dataset_subset, load_full_dataset


def preprocessing(example):
    """ minimal pre-processing applied to each example """
    example = example.strip()
    example = " ".join(example.split())
    return example


def dataset_preprocessing(version: str = 'full', subset_size: int = 1000):
    """ full dataset pre-processing"""
    if version == 'full':
        dataset = load_full_dataset(limit=50000)
    elif version == 'subset':
        dataset = load_dataset_subset(n_samples=subset_size)
    else:
        raise ValueError(f"Unknown version: {version} (expected 'full' or 'subset')")
    # we assume len(sentence) should be bigger than 2 to be long enough
    processed_dataset = [(pair['english'], pair['non_english']) for pair in dataset if len(pair['english'])>2 and len(pair['non_english'])>2]
    processed_dataset = [(preprocessing(s1), preprocessing(s2)) for s1, s2 in processed_dataset]
    return processed_dataset


