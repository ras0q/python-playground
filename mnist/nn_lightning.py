from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms
from torchvision.datasets import MNIST


class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, image_shape=(1, 28, 28), hidden_units=(32, 16)) -> None:
        super().__init__()
        # Lightningの新しい属性
        # self.train_acc = Accuracy(task="multiclass")
        # self.valid_acc = Accuracy(task="multiclass")
        # self.test_acc = Accuracy(task="multiclass")
        self.train_acc = MulticlassAccuracy(num_classes=10)
        self.valid_acc = MulticlassAccuracy(num_classes=10)
        self.test_acc = MulticlassAccuracy(num_classes=10)

        # nn.pyと同様のモデル
        input_size = image_shape[0] * image_shape[1] * image_shape[2]
        all_layers: list[nn.Module] = [nn.Flatten()]
        for hidden_unit in hidden_units:
            all_layers.append(nn.Linear(input_size, hidden_unit))
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
        all_layers.append(nn.Linear(input_size, 10))
        all_layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*all_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # training

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_training_epoch_end(self, outs: list[torch.Tensor]) -> None:
        self.log(
            "train_acc",
            self.train_acc.compute(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_acc.reset()

    # validation

    def validation_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log(
            "valid_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "valid_acc",
            self.valid_acc.compute(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.valid_acc.reset()

    # test

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_path="./") -> None:
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self) -> None:
        MNIST(root=self.data_path, download=True)

    # stage: Literal["fit", "validate", "test", "predict"]
    def setup(self, stage: str) -> None:
        mnist_all = MNIST(
            root=self.data_path, train=True, transform=self.transform, download=False
        )
        self.train, self.val = random_split(
            mnist_all, [55000, 5000], generator=torch.Generator().manual_seed(1)
        )
        self.test = MNIST(
            root=self.data_path, train=False, transform=self.transform, download=False
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=32, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=32, num_workers=4)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()


torch.manual_seed(1)

mnist_dm = MnistDataModule()
mnistclassifier = MultiLayerPerceptron()
trainer = pl.Trainer(max_epochs=20)

trainer.fit(mnistclassifier, mnist_dm)
