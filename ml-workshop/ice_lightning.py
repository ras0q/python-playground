import csv
from typing import List, Tuple

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split


class IceModel(pl.LightningModule):
    encoder: nn.Sequential
    lr: float
    loss_fn: nn.Module

    def __init__(self, encoder: nn.Sequential, lr: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        self.log(
            "valid_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def predict_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        x, *_ = batch
        return self.forward(x)


class DataModule(pl.LightningDataModule):
    batch_size: int
    num_workers: int
    train: Subset[Tuple[torch.Tensor, ...]]
    val: Subset[Tuple[torch.Tensor, ...]]
    test: Dataset

    def __init__(self, batch_size: int, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        train = pd.read_csv("https://www.abap34.com/ml-lecture/train.csv")
        test = pd.read_csv("https://www.abap34.com/ml-lecture/test.csv")
        train_x, train_y = train.drop(columns=["売り上げ"]), train["売り上げ"]

        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test = scaler.transform(test)
        train_y = train_y.values.reshape(-1, 1)

        train_val = TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        )
        val_size = int(len(train_val) * 0.3)
        self.train, self.val = random_split(
            train_val,
            [len(train_val) - val_size, val_size],
            generator=torch.Generator().manual_seed(34),
        )
        self.test = TensorDataset(torch.tensor(test, dtype=torch.float32))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    model = IceModel(
        encoder=nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ),
        lr=0.001,
    )
    dm = DataModule(batch_size=32, num_workers=4)

    trainer = pl.Trainer(max_epochs=20, log_every_n_steps=35)
    trainer.fit(model, datamodule=dm)

    preds = trainer.predict(model, datamodule=dm)
    assert preds is not None
    with open("submit.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([x.tolist() for pred in preds for x in pred])
