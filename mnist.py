import os
import numpy as np
from prometheus_client import Enum
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 128 if AVAIL_GPUS else 64


def deactivate_ema(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m._saved_running_mean, m.running_mean = m.running_mean, None
            m._saved_running_var, m.running_var = m.running_var, None


def activate_ema(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.running_mean = m._saved_running_mean
            m.running_var = m._saved_running_var


def calculate_estimates(model):
    with torch.no_grad():
        for data in tqdm(model.train_dataloader()):
            model(data[0])


class Modes(int, Enum):
    no_BN = 0
    basic_BN = 1
    almost_smart_BN = 2
    smart_BN = 3


class LitMNIST(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=8, learning_rate=5e-3, mode=Modes.smart_BN):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, _, _ = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        if mode == Modes.no_BN:
            self.model = nn.Sequential(
                nn.Conv2d(channels, hidden_size, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_size, self.num_classes, 3, 2, 1),
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(channels, hidden_size, 3, 2, 1),
                nn.BatchNorm2d(hidden_size, eps=1e-05, momentum=0.9,
                               affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
                nn.BatchNorm2d(hidden_size, eps=1e-05, momentum=0.9,
                               affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(hidden_size, self.num_classes, 3, 2, 1),
                nn.BatchNorm2d(self.num_classes, eps=1e-05, momentum=0.9,
                               affine=True, track_running_stats=True),
            )

        if mode == Modes.smart_BN or mode == Modes.almost_smart_BN:
            deactivate_ema(self.model)

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        x = x.sum(-1).sum(-1)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        return dict(loss=loss, acc=self.train_accuracy(preds, y))

    def training_epoch_end(self, training_step_outputs):
        loss, acc = [], []
        for out in training_step_outputs:
            loss.append(out["loss"].cpu().numpy())
            acc.append(out["acc"].cpu().numpy())

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("train_loss", np.mean(loss), prog_bar=True)
        self.log("train_acc", np.mean(acc), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True,
                               transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=1, shuffle=True)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("mode", type=int)
    args = parser.parse_args()

    if args.mode == 0:
        mode = Modes.no_BN
    elif args.mode == 1:
        mode = Modes.basic_BN
    elif args.mode == 2:
        mode = Modes.almost_smart_BN
    elif args.mode == 3:
        mode = Modes.smart_BN
    else:
        raise ValueError("Mode not allowed, select 0, 1, 2 or 3 for no batchnorm, " +
                         "basic batchnorm, almost smart & smart batchnorm respectively")

    model = LitMNIST(mode=mode)

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=10,
        progress_bar_refresh_rate=20,
    )

    trainer.fit(model)

    if mode == Modes.smart_BN or mode == Modes.almost_smart_BN:
        activate_ema(model)
        if mode == Modes.smart_BN:
            calculate_estimates(model)

    trainer.test(model)
