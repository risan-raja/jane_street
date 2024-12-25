import torch
from torch.utils.data import DistributedSampler
from typing import Optional
import math
import torch.distributed as dist
from ..datasets.js_dataset import JSTrainDataset


class JSPredictDataSampler(DistributedSampler):
    """
    JSPredictDataSampler is a custom distributed sampler for prediction datasets.
    This sampler ensures that each process in a distributed setting gets a unique subset of the dataset.
    It supports shuffling and dropping the last incomplete batch if necessary.
        dataset (JSTrainDataset): The dataset to sample from.
        num_replicas (Optional[int]): Number of processes participating in distributed training.
        rank (Optional[int]): Rank of the current process within num_replicas.
        shuffle (bool): If True, the sampler will shuffle the indices.
        seed (int): Random seed for shuffling.
        drop_last (bool): If True, drop the last incomplete batch.
    Attributes:
        dataset (JSTrainDataset): The dataset to sample from.
        index (Any): The index of the dataset.
        num_replicas (int): Number of processes participating in distributed training.
        rank (int): Rank of the current process within num_replicas.
        epoch (int): Current epoch number.
        drop_last (bool): If True, drop the last incomplete batch.
        shuffle (bool): If True, the sampler will shuffle the indices.
        seed (int): Random seed for shuffling.
        num_samples (int): Number of samples to draw for each process.
        total_size (int): Total size of the dataset after padding.
    Methods:
        __iter__(): Returns an iterator over the indices for the current process.
        __len__(): Returns the number of samples for the current process.
        set_epoch(epoch: int): Sets the epoch for this sampler to ensure different shuffling for each epoch.
    """

    def __init__(
        self,
        dataset: JSTrainDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        max_samples: int = 200000,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.index = dataset.sampler_index
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        if max_samples is not None and shuffle:
            self.index.sample(n=max_samples, seed=seed)
        elif max_samples is not None and not shuffle:
            self.index = self.index.head(n=max_samples)
        self.indexes = self.index["idx"].to_list()
        if self.drop_last and len(self.index) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.index) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.index) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(
                len(self.index["idx"].to_list()), generator=g
            ).tolist()  # type: ignore[arg-type]
            indices = self.index["idx"].to_numpy()[indices].ravel().tolist()
        else:
            indices = self.index["idx"].to_list()  # type: ignore[arg-type]
        # indices = self.index["idx"].to_list()
        # if not self.drop_last:
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
