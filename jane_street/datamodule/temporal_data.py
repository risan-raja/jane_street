import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from ..datasets import JSTSDataset, JSDatasetMeta, custom_collate_fn
from ..samplers import JSPredictDataSampler
from ..samplers.train_sampler_uni import JSTrainDataSampler


class JSDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super(JSDataModule, self).__init__()
        self.config = config
        self.train_index_path = config.train.index_path
        self.val_index_path = config.val.index_path
        self.test_index_path = config.test.index_path
        self.train_batch_size = config.train.batch_size
        self.val_batch_size = config.val.batch_size
        self.test_batch_size = config.test.batch_size
        self.n_cardinal = config.train.n_cardinal

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset_metadata = JSDatasetMeta(
                index_path=self.train_index_path
            )
            self.train_dataset = JSTSDataset(self.train_dataset_metadata)
            self.val_dataset_metadata = JSDatasetMeta(index_path=self.val_index_path)
            self.val_dataset = JSTSDataset(self.val_dataset_metadata)
        if stage == "test":
            self.test_dataset_metadata = JSDatasetMeta(index_path=self.test_index_path)
            self.test_dataset = JSTSDataset(self.test_dataset_metadata)

    def train_dataloader(self):
        self.train_dataset_metadata = JSDatasetMeta(index_path=self.train_index_path)
        self.train_dataset = JSTSDataset(self.train_dataset_metadata)
        self.train_sampler = JSTrainDataSampler(
            self.train_dataset,
            batch_size=self.train_batch_size,
            n_cardinal=self.n_cardinal,
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            sampler=self.train_sampler,
            collate_fn=custom_collate_fn,
            num_workers=24,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        self.val_dataset_metadata = JSDatasetMeta(index_path=self.test_index_path)
        self.val_dataset = JSTSDataset(self.val_dataset_metadata)
        self.val_sampler = JSPredictDataSampler(
            self.val_dataset,
            max_samples=self.config.val_dataset_size,
            shuffle=self.config.val_shuffle,
        )
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            sampler=self.val_sampler,
            collate_fn=custom_collate_fn,
            pin_memory=False,
            num_workers=16,
            drop_last=False,
        )

    def test_dataloader(self):
        self.test_dataset_metadata = JSDatasetMeta(index_path=self.test_index_path)
        self.test_dataset = JSTSDataset(self.test_dataset_metadata)
        self.test_sampler = JSPredictDataSampler(self.test_dataset, shuffle=False)
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            sampler=self.test_sampler,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            num_workers=8,
            drop_last=False,
        )

    def predict_dataloader(self):
        self.test_sampler = JSPredictDataSampler(self.test_dataset, shuffle=False)
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            sampler=self.test_sampler,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            num_workers=8,
            drop_last=False,
        )
