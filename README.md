# Problem-of-BatchNorm

This public repository is a simple toy model (MNIST) training script based on Pytorch Lightning.
The goal is to highlight the different behaviours of BatchNorm layers.
It comes with functions to plot the evolution of the metrics for the 4 modes:

- Mode 0: No BatchNorm layers are used
- Mode 1: Basic BatchNorm (no modifications)
- Mode 2: Almost Smart BatchNorm (We activate the running stats for inference but we don’t run the model 1 epoch to estimate the moving average of stats)
- Mode 3: Smart BatchNorm (We estimate on 1 epoch the average stats of the dataset before inference mode)

To train the model with the 4 distinctive modes, just execute in CLI (python 3.8):

```
pip install -r requirements.txt
python mnist.py mode
```

With mode = 0, 1, 2 or 3
