In this repo, we replicate the paper of [Nils Reimers, Iryna Gurevych.](https://arxiv.org/abs/2004.09813) under resource constraints: compact teacher (MiniLM) and limited data (50k EN-DE) sentences. Our paper with results is available [here](Tsanda_Mitsiulia_paper.pdf).

To train a student model with knowledge distillation, configure hyperparameters in configs/sample.yaml and run main.py. Weights are saved to models/checkpoints.

_NOTE:_ Evaluation scripts require a specific datasets version:

<pre> pip install datasets==2.19.0 </pre>


