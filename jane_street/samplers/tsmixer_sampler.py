from torch.utils.data import DistributedSampler
from typing import Optional
import polars as pl
import torch.distributed as dist
from ..datasets.js_tsdataset import JSTSDataset
import numpy as np
import math

pl.enable_string_cache()


class JSMixerTrainDataSampler(DistributedSampler):
    """
    JSMixerTrainDataSampler is a sampler class designed to handle distributed training data
    for time-series mixing tasks in a PyTorch-based workflow. It inherits from DistributedSampler
    and provides the following functionality:
    • Manages and balances subsets of a JSTSDataset across multiple replicas for distributed
        data parallel training.
    • Randomly shuffles indices across epochs while ensuring each replica receives a unique
        subset of data.
    • Allows setting a specific epoch to control the sampling seed for reproducible random
        ordering.
    • Dynamically groups and slices dataset indices based on certain criteria (e.g.,
        decoder_length, filter conditions) to form mini-batches.
    • Ensures indexes returned are properly partitioned for multiple replicas in a distributed
        environment and can optionally drop incomplete batches.
            dataset (JSTSDataset): The dataset to sample from.
            num_replicas (int, optional): Number of processes participating in distributed training.
                    If None, it will be obtained from PyTorch distributed package.
            rank (int, optional): Rank of the current process within num_replicas.
                    If None, it will be obtained from PyTorch distributed package.
            shuffle (bool): If True, the sampler will shuffle the indices before returning them.
            seed (int): Random seed for shuffling the indices.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If True, drops the last incomplete batch if its size would be less
                    than batch_size.
            n_cardinal (int): Additional multiplier for the number of samples processed per epoch.
    Attributes:
            dataset (JSTSDataset): The dataset object from which samples are drawn.
            mindex (pl.DataFrame): Polars DataFrame containing index and filter logic for data sampling.
            grouped_index (pl.DataFrame): Groups and slices the indices based on decoder_length
                    and other conditions.
            batch_size (int): Number of samples per batch.
            num_replicas (int): Number of processes in distributed training.
            rank (int): The rank of the current process.
            epoch (int): Current epoch counter ensuring a different random ordering for each epoch.
            drop_last (bool): Whether to drop the last incomplete batch.
            shuffle (bool): Indicates if the sampler should shuffle the indices.
            seed (int): Random seed used for shuffling.
            n_cardinal (int): Multiplier controlling how many samples are processed per epoch.
            total_size (int): Computed total size of samples for all replicas.
            num_samples (int): Number of samples allocated to each replica.
    Usage:
            sampler = JSMixerTrainDataSampler(dataset, num_replicas=4, rank=0, shuffle=True)
            for index in sampler:
                    # process data index
    Note:
            This sampler is particularly useful when working in a distributed environment
            where multiple processes read from the same dataset and coordinate sampling
            to avoid overlapping subsets of data.
    """

    def __init__(
        self,
        dataset: JSTSDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 212,
        batch_size: int = 64,
        drop_last: bool = True,
        n_cardinal: int = 2,
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
        self.mindex = dataset.sampler_index
        self.max_g = 32776 // batch_size * n_cardinal * num_replicas
        self.grouped_index = (
            self.mindex.filter(pl.col("start_date").cast(pl.Int32) >= 677)
            .group_by("decoder_length")
            .agg(
                pl.col("idx"),
            )
            .with_columns(
                pl.col("decoder_length").cast(pl.Int32),
                pl.col("idx").list.len().alias("count"),
            )
            .with_columns(
                pl.col("idx")
                .list.slice(0, batch_size * n_cardinal * num_replicas * self.max_g)
                .alias("idx")
            )
        )
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.n_cardinal = n_cardinal
        self.total_size = self.batch_size * 968 * self.n_cardinal * self.num_replicas
        self.num_samples = self.total_size // self.num_replicas

    @property
    def index(self):
        indices = (
            self.grouped_index.sample(n=968, shuffle=True, seed=self.seed + self.epoch)
            .select(
                pl.col("idx").list.slice(
                    self.epoch * self.batch_size * self.num_replicas * self.n_cardinal,
                    (self.epoch + 1)
                    * self.batch_size
                    * self.num_replicas
                    * self.n_cardinal,
                )
            )["idx"]
            .to_numpy()
            .ravel()
        )
        return np.concatenate(indices).astype("int").tolist()

    def __iter__(self):
        indices = self.index
        indices = indices[self.rank : self.total_size : self.num_replicas][
            : self.num_samples
        ]

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class JSMixerValDataSampler(DistributedSampler):
    """
    A PyTorch DistributedSampler for validation data in the JSMixer model.
    This sampler is designed to work with distributed training setups, ensuring that each process
    gets a unique subset of the validation data sequentially.

    Attributes:
        dataset (JSTSDataset): The dataset to sample from.
        mindex (pl.DataFrame): The index of the dataset.
        max_g (int): Maximum number of groups.
        grouped_index (pl.DataFrame): Grouped index of the dataset.
        batch_size (int): Number of samples per batch.
        num_replicas (int): Number of processes participating in distributed training.
        rank (int): Rank of the current process within num_replicas.
        epoch (int): Current epoch.
        drop_last (bool): If True, drop the last incomplete batch.
        shuffle (bool): If True, the sampler will shuffle the indices.
        seed (int): Random seed for shuffling.
        n_cardinal (int): Cardinality of the data.
        total_size (int): Total size of the dataset.
        num_samples (int): Number of samples per replica.
    Methods:
        index: Returns the indices for the current epoch.
        __iter__: Returns an iterator over the indices for the current epoch.
        __len__: Returns the number of samples per replica.
        set_epoch: Sets the epoch for this sampler.
    """

    def __init__(
        self,
        dataset: JSTSDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 212,
        batch_size: int = 64,
        drop_last: bool = True,
        n_cardinal: int = 2,
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
        self.mindex = dataset.sampler_index
        self.max_samples = len(self.mindex)
        self.num_replicas = num_replicas
        if self.max_samples % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.max_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.max_samples / self.num_replicas)  # type: ignore[arg-type]
        self.max_g = 3765 // batch_size * n_cardinal * num_replicas
        self.grouped_index = (
            self.mindex.filter(pl.col("start_date").cast(pl.Int32) >= 677)
            .group_by("decoder_length")
            .agg(
                pl.col("idx"),
            )
            .with_columns(
                pl.col("decoder_length").cast(pl.Int32),
                pl.col("idx").list.len().alias("count"),
            )
            .with_columns(
                pl.col("idx")
                .list.slice(0, batch_size * n_cardinal * num_replicas * self.max_g)
                .alias("idx")
            )
            .sort("decoder_length")
            .with_columns(pl.col("idx").list.sort(descending=False))
        )
        self.batch_size = batch_size
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = False
        self.seed = seed
        self.n_cardinal = n_cardinal
        self.total_size = (
            self.batch_size * 968 * self.n_cardinal * self.num_replicas * self.max_g
        )
        self.num_samples = self.total_size // self.num_replicas

    @property
    def index(self):
        indices = (
            self.grouped_index.select(
                pl.col("idx").list.slice(
                    # (self.epoch // 2)
                    # * self.batch_size
                    # * self.num_replicas
                    # * self.n_cardinal,
                    # (self.epoch // 2 + 1)
                    # * self.batch_size
                    # * self.num_replicas
                    # * self.n_cardinal,
                    -self.batch_size * self.n_cardinal * self.num_replicas * self.max_g,
                    None,
                )
            )
            .to_numpy()
            .ravel()
        )
        return np.concatenate(indices).astype("int").tolist()

    def __iter__(self):
        indices = self.index
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        indices = indices[self.rank : self.total_size : self.num_replicas][
            : self.num_samples
        ]

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
