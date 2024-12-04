import os
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from .mixed_dataset import MixedDataset


class MixedDataModule(LightningDataModule):
    def __init__(
        self, bert_model, dataset_path, tool_capacity, batch_size, num_workers, seed
    ):
        super().__init__()
        self.bert_model = bert_model
        self.dataset_path = dataset_path
        self.tool_capacity = tool_capacity
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = MixedDataset(
                self.bert_model,
                "train",
                os.path.join(self.dataset_path, "train.json"),
                self.tool_capacity,
                seed=self.seed,
            )
            self.val_dataset = MixedDataset(
                self.bert_model,
                "test",
                os.path.join(self.dataset_path, "test.json"),
                self.tool_capacity,
                seed=self.seed,
            )
        elif stage == "test":
            self.test_dataset = MixedDataset(
                self.bert_model,
                "test",
                os.path.join(self.dataset_path, "test.json"),
                self.tool_capacity,
                seed=self.seed,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
