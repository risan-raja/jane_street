from typing import Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import math


def get_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).

    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.

    Returns:
        int: embedding size
    """
    if n > 2:
        return min(round(1.6 * n**0.56), max_size)
    else:
        return 1


class TimeDistributedEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, *args, batch_first: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x):  # type: ignore
        if len(x.size()) <= 2:
            return super().forward(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        y = super().forward(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class MultiEmbedding(nn.Module):
    concat_output: bool

    def __init__(
        self,
        embedding_sizes: Union[
            Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]]
        ],
        x_categoricals: Optional[List[str]] = None,
        categorical_groups: Optional[Dict[str, List[str]]] = None,
        categorical_groups_name_index: Optional[Dict[str, int]] = None,
        embedding_paddings: Optional[List[str]] = None,
        max_embedding_size: Optional[int] = None,
    ):
        """Embedding layer for categorical variables including groups of categorical variables.

        Enabled for static and dynamic categories (i.e. 3 dimensions for batch x time x categories).

        Args:
            embedding_sizes (Union[Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]]]):
                either

                * dictionary of embedding sizes, e.g. ``{'cat1': (10, 3)}``
                  indicates that the first categorical variable has 10 unique values which are mapped to 3 embedding
                  dimensions. Use :py:func:`~pytorch_forecasting.utils.get_embedding_size` to automatically obtain
                  reasonable embedding sizes depending on the number of categories.
                * dictionary of categorical sizes, e.g. ``{'cat1': 10}`` where embedding sizes are inferred by
                  :py:func:`~pytorch_forecasting.utils.get_embedding_size`.
                * list of embedding and categorical sizes, e.g. ``[(10, 3), (20, 2)]`` (requires ``x_categoricals`` to
                  be empty)
                * list of categorical sizes where embedding sizes are inferred by
                  :py:func:`~pytorch_forecasting.utils.get_embedding_size` (requires ``x_categoricals`` to be empty).

                If input is provided as list, output will be a single tensor of shape batch x (optional) time x
                sum(embedding_sizes). Otherwise, output is a dictionary of embedding tensors.
            x_categoricals (List[str]): list of categorical variables that are used as input.
            categorical_groups (Dict[str, List[str]]): dictionary of categories that should be summed up in an
                embedding bag, e.g. ``{'cat1': ['cat2', 'cat3']}`` indicates that a new categorical variable ``'cat1'``
                is mapped to an embedding bag containing the second and third categorical variables.
                Defaults to empty dictionary.
            embedding_paddings (List[str]): list of categorical variables for which the value 0 is mapped to a zero
                embedding vector. Defaults to empty list.
            max_embedding_size (int, optional): if embedding size defined by ``embedding_sizes`` is larger than
                ``max_embedding_size``, it will be constrained. Defaults to None.
        """
        if categorical_groups is None:
            categorical_groups = {}
        if embedding_paddings is None:
            embedding_paddings = []
        super().__init__()
        if isinstance(embedding_sizes, dict):
            self.concat_output = False  # return dictionary of embeddings
            # conduct input data checks
            assert x_categoricals is not None, "x_categoricals must be provided."
            categorical_group_variables = [
                name for names in categorical_groups.values() for name in names
            ]
            if len(categorical_groups) > 0:
                assert all(
                    name in embedding_sizes for name in categorical_groups
                ), "categorical_groups must be in embedding_sizes."
                assert not any(
                    name in embedding_sizes for name in categorical_group_variables
                ), "group variables in categorical_groups must not be in embedding_sizes."
                assert all(
                    name in x_categoricals for name in categorical_group_variables
                ), "group variables in categorical_groups must be in x_categoricals."
            assert all(
                name in embedding_sizes
                for name in embedding_sizes
                if name not in categorical_group_variables
            ), (
                "all variables in embedding_sizes must be in x_categoricals - but only if"
                "not already in categorical_groups."
            )
        else:
            assert (
                x_categoricals is None and len(categorical_groups) == 0
            ), "If embedding_sizes is not a dictionary, categorical_groups and x_categoricals must be empty."
            # number embeddings based on order
            embedding_sizes = {
                str(name): size for name, size in enumerate(embedding_sizes)
            }  # type: ignore
            if isinstance(embedding_sizes, dict):
                x_categoricals = list(embedding_sizes.keys())
            self.concat_output = True

        # infer embedding sizes if not determined
        if isinstance(embedding_sizes, dict):
            self.embedding_sizes = {
                name: (size, get_embedding_size(size))
                if isinstance(size, int)
                else size
                for name, size in embedding_sizes.items()  # type: ignore
            }
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.max_embedding_size = max_embedding_size
        self.x_categoricals = x_categoricals
        if isinstance(x_categoricals, list):
            self.x_categoricals_index = {
                name: i for i, name in enumerate(x_categoricals)
            }
        else:
            self.x_categoricals_index = {}
        self.categorical_groups_name_index = categorical_groups_name_index
        self.init_embeddings()

    def init_embeddings(self):
        self.embeddings = nn.ModuleDict()
        for name in self.embedding_sizes.keys():
            embedding_size = self.embedding_sizes[name][1]
            if self.max_embedding_size is not None:
                embedding_size = min(embedding_size, self.max_embedding_size)
            # convert to list to become mutable
            # type: ignore
            self.embedding_sizes[name]: list = list(self.embedding_sizes[name])  # type: ignore
            self.embedding_sizes[name][1] = embedding_size  # type: ignore
            if name in self.categorical_groups:  # embedding bag if related embeddings
                self.embeddings[name] = TimeDistributedEmbeddingBag(
                    self.embedding_sizes[name][0],
                    embedding_size,
                    mode="sum",
                    batch_first=True,
                )
            else:
                if name in self.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.embeddings[name] = nn.Embedding(
                    self.embedding_sizes[name][0],
                    embedding_size,
                    padding_idx=padding_idx,
                )

    def names(self):
        return list(self.keys())

    def items(self):
        return self.embeddings.items()

    def keys(self):
        return self.embeddings.keys()

    def values(self):
        return self.embeddings.values()

    def __getitem__(self, name: str):
        return self.embeddings[name]

    @property
    def input_size(self) -> int:
        return len(self.x_categoricals)  # type: ignore

    @property
    def output_size(self) -> Union[Dict[str, int], int]:
        if self.concat_output:
            return sum([s[1] for s in self.embedding_sizes.values()])
        else:
            return {name: s[1] for name, s in self.embedding_sizes.items()}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input tensor of shape batch x (optional) time x categoricals in the order of
                ``x_categoricals``.

        Returns:
            Union[Dict[str, torch.Tensor], torch.Tensor]: dictionary of category names to embeddings
                of shape batch x (optional) time x embedding_size if ``embedding_size`` is given as dictionary.
                Otherwise, returns the embedding of shape batch x (optional) time x sum(embedding_sizes).
                Query attribute ``output_size`` to get the size of the output(s).
        """
        input_vectors = {}
        for name, emb in self.embeddings.items():
            if name in self.categorical_groups:
                # print(self.categorical_groups_name_index[name])
                input_vectors[name] = emb(
                    x[
                        ...,
                        [0, 1, 2],
                    ]
                )
            else:
                input_vectors[name] = emb(x[..., self.x_categoricals_index[name]])  # type: ignore

        if self.concat_output:  # concatenate output
            return torch.cat(list(input_vectors.values()), dim=-1)  # type: ignore
        else:
            return input_vectors


class TemporalPosEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2
        self.div_term = torch.exp(
            torch.arange(0, half_dim, 2) * (-math.log(10000.0) / half_dim)
        )

    def forward(self, date_id, time_idx):
        # date_id and time_idx should be tensors of shape (batch_size, seq_len, 1)

        # Create separate embeddings for date_id
        # 1. Expand div_term to be compatible with date_id
        div_term_date = (
            self.div_term.unsqueeze(0).unsqueeze(0).to(date_id.device)
        )  # Shape becomes (1, 1, half_dim // 2)

        # 2. unsqueeze date_id to align with div_term_date and perform element-wise multiplication
        date_vals = date_id * div_term_date  # Broadcasting happens here

        # 3. Create embeddings
        date_embeddings = torch.zeros(
            date_id.shape[0], date_id.shape[1], self.embedding_dim // 2
        ).to(date_id.device)
        date_embeddings[:, :, 0::2] = torch.sin(date_vals)
        date_embeddings[:, :, 1::2] = torch.cos(date_vals)

        # Create separate embeddings for time_idx
        # 1. Expand div_term to be compatible with time_idx
        div_term_time = self.div_term.unsqueeze(0).unsqueeze(0).to(time_idx.device)

        # 2. Unsqeeze time_idx to align with div_term_time and perform element-wise multiplication
        time_vals = time_idx * div_term_time

        # 3. Create embeddings
        time_embeddings = torch.zeros(
            time_idx.shape[0], time_idx.shape[1], self.embedding_dim // 2
        ).to(time_idx.device)
        time_embeddings[:, :, 0::2] = torch.sin(time_vals)
        time_embeddings[:, :, 1::2] = torch.cos(time_vals)
        return date_embeddings, time_embeddings


class TemporalEmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.date_embedding = nn.Linear(embedding_dim // 2, embedding_dim // 2)
        self.time_embedding = nn.Linear(embedding_dim // 2, embedding_dim // 2)

    def forward(self, date_id, time_idx):
        date_emb = self.date_embedding(date_id)
        time_emb = self.time_embedding(time_idx)
        return torch.cat([date_emb, time_emb], dim=-1)
