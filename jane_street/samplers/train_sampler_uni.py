import torch
from torch.utils.data import DistributedSampler
from typing import Optional
import polars as pl
import torch.distributed as dist
from ..datasets.js_tsdataset import JSTSDataset
import numpy as np
import random

pl.enable_string_cache()


class JSTrainDataSampler(DistributedSampler):
    """
    JSTrainDataSampler is a custom distributed sampler for prediction datasets.
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
        dataset: JSTSDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
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
        self.grouped_index = self.mindex.group_by(["end_date", "decoder_length"]).agg(
            pl.col("symbol_id"), pl.col("idx")
        )
        self.date_min = self.mindex["end_date"].cast(pl.Int16).min()
        self.date_max = self.mindex["end_date"].cast(pl.Int16).max()
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
        print(
            f"num_samples {self.num_samples} total_size {self.total_size} rank {self.rank} num_replicas {self.num_replicas}"
        )

    @property
    def index(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        tsteps = [np.arange(1, 969) for _ in range(self.n_cardinal)]
        tsteps = np.concatenate(tsteps)
        perm = torch.randperm(len(tsteps), generator=g).numpy()
        tsteps = tsteps[perm]
        batch_dates = []
        for tstep_idx in tsteps:
            g.manual_seed(self.seed + self.epoch + int(tstep_idx))
            if tstep_idx < 848:
                dates = torch.randint(
                    self.date_min,
                    self.date_max + 1,
                    (self.batch_size * self.num_replicas,),
                    generator=g,
                ).numpy()
            else:
                dates = torch.randint(
                    677,
                    self.date_max + 1,
                    (self.batch_size * self.num_replicas,),
                    generator=g,
                ).numpy()
            batch_dates.extend(dates)
        tsteps = np.repeat(tsteps, self.batch_size * self.num_replicas)
        with pl.StringCache():
            ci = pl.DataFrame(
                {
                    "end_date": [str(x) for x in batch_dates],
                    "decoder_length": [str(x) for x in tsteps],
                }
            )
            ci = ci.with_columns(
                pl.col("end_date").cast(pl.Categorical),
                pl.col("decoder_length").cast(pl.Categorical),
            )
            ci = ci.join(
                self.grouped_index, on=["end_date", "decoder_length"], how="inner"
            )
        rb = ci.map_rows(lambda df: random.choice(df[2])).to_numpy().ravel()
        ci = ci.with_columns(pl.lit(rb).alias("sel_symbol_id"))
        res_i = ci.map_rows(lambda li: li[3][li[2].index(li[4])]).to_numpy().ravel()
        res_ = (
            ci.with_columns(pl.lit(res_i).alias("sel_idx"))
            .select("sel_idx")["sel_idx"]
            .to_list()
        )
        return res_

    def __iter__(self):
        indices = self.index
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
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
