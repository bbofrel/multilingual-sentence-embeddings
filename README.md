**Welcome to the repository of the knowledge-distillation project.**
To train your student model with knowledge distillation from a teacher model, you need to set up your environment, specify hyperparameters in configs/sample.yaml and run main.py. Your model's weights will be saved in models/checkpoints.

_NOTE:_ to run evaluation, you need to run the respective scripts directly from the evaluation folder. Besides, you need the datasets version of 2.19.0 for doing so. We suggest that you first train the model using the requirements file in this repository, and then run "pip install datasets==2.19.0" for evaluation specifically.
