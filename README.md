**Welcome to the repository of the knowledge-distillation project.**

In this repo, we replicate the paper of [Nils Reimers, Iryna Gurevych.](https://arxiv.org/abs/2004.09813) You can find our paper with the obtained results [here](Tsanda_Mitsiulia_paper.pdf).

To train your student model with knowledge distillation from a teacher model, you need to set up your environment, specify hyperparameters in configs/sample.yaml and run main.py. Your model's weights will be saved in models/checkpoints.

_NOTE:_ to run evaluation, you need to run the respective scripts directly from the evaluation folder. Besides, you need the datasets version of 2.19.0 for doing so. We suggest that you first train the model using the requirements file in this repository, and then run the following specifically for the evaluation:

<pre> ```pip install datasets==2.19.0 ``` </pre>


