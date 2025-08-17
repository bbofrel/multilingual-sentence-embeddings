import yaml
from training.train_student import prepare_dataset


def main():
    with open("configs/sample.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataloader = prepare_dataset(config)
    print(dataloader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
