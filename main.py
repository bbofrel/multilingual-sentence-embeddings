import yaml
from training.train_student import *


def main():
    with open("configs/sample.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_loader, dev_loader = prepare_dataset(config)
    print("dataset is loaded")
    training_loop(config, train_loader, dev_loader)
    print("training started")


if __name__ == '__main__':
    main()
