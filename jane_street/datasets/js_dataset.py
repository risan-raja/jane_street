from .metadata import JSDatasetMeta
import torch
from torch.utils.data import Dataset
from torch.nn.utils import rnn
import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Any


class JSTrainDataset(Dataset):
    def __init__(self, dataset_metadata: JSDatasetMeta, hist_concat=False, strict=True):
        super(JSTrainDataset, self).__init__()
        self.metadata = dataset_metadata
        self.responder_vars_idx = [84, 85, 86, 87, 88, 89, 90, 91, 92]
        self.categorical_vars_idx = [2, 3, 14, 15, 16]
        self.weight_idx = [4]
        self.time_vars_idx = [0, 1, 2]
        self.symbol_vars_idx = 3
        self.static_covariate_idx = [3]
        self.target_idx = [89]
        self.feature_vars_idx = [
            i
            for i in range(93)
            if i
            not in self.responder_vars_idx
            + self.categorical_vars_idx
            + self.time_vars_idx
            + [self.symbol_vars_idx]
            + self.target_idx
        ]
        self.hist_concat = hist_concat
        self.strict = strict
        # Feature 09, 10, 11 are ordinal categorical variables with variable scale
        # Encoding them with OrdinalEncoder
        self.categorical_encoder = ColumnTransformer(
            transformers=[
                (
                    "feature_09",
                    OrdinalEncoder(
                        categories=[self.metadata.categories["feature_09"]],
                        handle_unknown="use_encoded_value",
                        unknown_value=len(self.metadata.categories["feature_09"]),
                        dtype=np.int16,
                    ),
                    [2],
                ),
                (
                    "feature_10",
                    OrdinalEncoder(
                        categories=[self.metadata.categories["feature_10"]],
                        handle_unknown="use_encoded_value",
                        unknown_value=len(self.metadata.categories["feature_10"]),
                        dtype=np.int16,
                    ),
                    [3],
                ),
                (
                    "feature_11",
                    OrdinalEncoder(
                        categories=[self.metadata.categories["feature_11"]],
                        handle_unknown="use_encoded_value",
                        unknown_value=len(self.metadata.categories["feature_11"]),
                        dtype=np.int16,
                    ),
                    [4],
                ),
            ],
            remainder="passthrough",
        )
        self.feature_scaler = self.metadata.feature_scaler
        self.index_keys = {
            "symbol_id": 0,
            "start_date": 1,
            "end_date": 2,
            "encoder_length": 3,
            "decoder_length": 4,
        }
        self.index = self.metadata.index
        self.sampler_index = self.metadata.sampler_index
        self.root = self.metadata.root
        self.real_scaler = StandardScaler()

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _to_tensor(data: Any, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype)

    def __getitem__(self, idx):
        index = self.index[idx].to_numpy().ravel()
        symbol_id = int(index[self.index_keys["symbol_id"]])
        start_date = int(index[self.index_keys["start_date"]])
        end_date = int(index[self.index_keys["end_date"]])
        encoder_length = self._to_tensor(
            int(index[self.index_keys["encoder_length"]]), torch.int16
        )
        decoder_length = self._to_tensor(
            int(index[self.index_keys["decoder_length"]]), torch.int16
        )
        symbol_hist_data: ArrayLike = self.root[f"{symbol_id}/{start_date}"][:]  # type: ignore
        symbol_curr_data: ArrayLike = self.root[f"{symbol_id}/{end_date}"][:][
            :decoder_length, :
        ]  # type: ignore
        static_covariates = self._to_tensor(np.array([symbol_id]), torch.int16).long()
        encoder_reals: ArrayLike = self._to_tensor(
            self.real_scaler.fit_transform(
                symbol_hist_data[:, self.weight_idx + self.feature_vars_idx]  # type: ignore
            ),  # type: ignore
            torch.float32,
        )
        encoder_targets = self._to_tensor(
            symbol_hist_data[:, self.responder_vars_idx],  # type: ignore
            torch.float32,  # type: ignore
        )
        encoder_categoricals = self._to_tensor(
            self.categorical_encoder.fit_transform(
                symbol_hist_data[:, self.categorical_vars_idx]  # type: ignore
            ),
            dtype=torch.int16,
        ).long()
        decoder_reals = self._to_tensor(
            self.real_scaler.transform(
                symbol_curr_data[:, self.weight_idx + self.feature_vars_idx]  # type: ignore
            ),
            torch.float32,
        )
        decoder_categoricals = self._to_tensor(
            self.categorical_encoder.fit_transform(
                symbol_curr_data[:, self.categorical_vars_idx]  # type: ignore
            ),
            torch.int16,
        ).long()
        targets = self._to_tensor(symbol_curr_data[:, self.target_idx], torch.float32)  # type: ignore
        weights = self._to_tensor(symbol_curr_data[:, self.weight_idx], torch.float32)  # type: ignore
        # Data is returned as (num_samples, time_steps, features)
        return (
            {
                "encoder_length": encoder_length,
                "decoder_length": decoder_length,
                "static_covariates": static_covariates,
                "encoder_reals": encoder_reals,
                "encoder_categoricals": encoder_categoricals,
                "encoder_targets": encoder_targets,
                "decoder_reals": decoder_reals,
                "decoder_categoricals": decoder_categoricals,
                "target_scale": torch.tensor([-5, 5], dtype=torch.float32),
            },
            (targets, weights),
        )


def custom_collate_fn(
    batches: list[tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]],
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    return (
        dict(
            encoder_lengths=torch.tensor(
                [batch[0]["encoder_length"] for batch in batches], dtype=torch.long
            ),
            decoder_lengths=torch.tensor(
                [batch[0]["decoder_length"] for batch in batches], dtype=torch.long
            ),
            groups=torch.stack([batch[0]["static_covariates"] for batch in batches]),
            encoder_reals=rnn.pad_sequence(
                [batch[0]["encoder_reals"] for batch in batches], batch_first=True
            ),
            encoder_categoricals=rnn.pad_sequence(
                [batch[0]["encoder_categoricals"] for batch in batches],
                batch_first=True,
            ),
            decoder_reals=rnn.pad_sequence(
                [batch[0]["decoder_reals"] for batch in batches], batch_first=True
            ),
            decoder_categoricals=rnn.pad_sequence(
                [batch[0]["decoder_categoricals"] for batch in batches],
                batch_first=True,
            ),
            encoder_targets=rnn.pad_sequence(
                [batch[0]["encoder_targets"] for batch in batches], batch_first=True
            ),
            target_scale=torch.stack([batch[0]["target_scale"] for batch in batches]),
        ),
        (
            rnn.pad_sequence([batch[1][0] for batch in batches], batch_first=True),
            rnn.pad_sequence([batch[1][1] for batch in batches], batch_first=True),
        ),
    )
