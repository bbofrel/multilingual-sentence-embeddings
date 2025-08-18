import yaml
from training.train_student import *


def main():
    with open("configs/sample.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataloader = prepare_dataset(config)
    print(dataloader)
    batch = next(iter(dataloader))
    print(batch.keys())
    training_loop(config, dataloader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
