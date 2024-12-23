import torch
from torch import Tensor
from torch.utils.data import DistributedSampler
from typing import Optional
import polars as pl
import math
import torch.distributed as dist
from ..datasets.js_dataset import JSTrainDataset


class JSTrainDistributedSampler(DistributedSampler):
    """
    A custom distributed sampler for training jobs that extends PyTorch's DistributedSampler.
    This sampler divides the dataset across multiple processes for efficient distributed training.
    It ensures that each process sees a unique subset of the original dataset in each epoch.
    Attributes:
        dataset (JSTrainDataset): The dataset to sample from.
        frequency (int): Specifies how often certain operations or checks are performed per epoch.
        num_replicas (int): Number of parallel processes participating in distributed training.
        rank (int): The rank (or index) of the current process among all replicas.
        shuffle (bool): Indicates whether the data should be shuffled at the start of each epoch.
        seed (int): The random seed used for shuffling.
        drop_last (bool): If True, drops the last incomplete batch during an epoch, if it does not
            evenly divide the dataset size.
    Raises:
        RuntimeError: If the PyTorch distributed package (dist) is not available.
        ValueError: If the provided rank is outside the valid range for the number of replicas.
    Example:
        >>> sampler = JSTrainDistributedSampler(dataset=my_dataset, num_replicas=4, rank=0)
        >>> loader = DataLoader(my_dataset, sampler=sampler, batch_size=32)
    """

    def __init__(
        self,
        dataset: JSTrainDataset,
        frequency: int = 3,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
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
        # Since the previous day is used for prediction the data from the next
        # Day might creep into the same epoch. To avoid this we skip a day.
        self.frequency = frequency
        self.shuffle = shuffle
        self.seed = seed

    @property
    def date_skip(self) -> int:
        return self.epoch % self.frequency

    @property
    def epoch_dates(self) -> Tensor:
        # print(self.index["end_date"].cast(pl.Int16).min(), self.index["end_date"].cast(pl.Int16).max())
        date_min: str = self.index["end_date"].cast(pl.Int16).min()  # type: ignore
        date_max: str = self.index["end_date"].cast(pl.Int16).max()  # type: ignore
        return torch.arange(
            int(date_min) + self.date_skip,
            int(date_max) + 1,
            step=self.frequency,
            dtype=torch.int16,
        )

    def assign_time_steps(self, dates, g) -> Tensor:
        time_steps = torch.arange(1, 969)[torch.randperm(968, generator=g)][
            : dates.shape[0]
        ]
        time_steps[dates < 677] = time_steps[dates < 677].clip(1, 849)
        return time_steps

    @staticmethod
    def stringify(tensor: Tensor) -> list[str]:
        return [str(elem.item()) for elem in tensor]

    def get_indices_epoch(self, g):
        print("Accessed")
        dates = self.epoch_dates
        time_steps = self.stringify(self.assign_time_steps(dates, g))
        dates = self.stringify(dates)
        indices = []
        for date, time_step in zip(dates, time_steps):
            indices.extend(
                self.index.filter(
                    (self.index["end_date"] == date)
                    & (self.index["decoder_length"] == time_step)
                )["idx"].to_list()
            )
        print("Indices Generated")
        return indices

    @staticmethod
    def upper_limit(date):
        return 968 if date > 676 else 849

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = self.get_indices_epoch(g)  # type: ignore[arg-type]
        else:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = self.get_indices_epoch(g)  # type: ignore[arg-type]

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(indices) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(indices) / self.num_replicas)  # type: ignore[arg-type]

        self.total_size = self.num_samples * self.num_replicas

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        print("Indices Assigned ", len(indices))
        return iter(indices)

    def __len__(self):
        return super().__len__()
