from .metadata import JSDatasetMeta
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Any


class JSTSDataset(Dataset):
    def __init__(
        self,
        dataset_metadata: JSDatasetMeta,
        hist_concat=False,
        strict=True,
        lookback_sampling=4,
    ):
        super(JSTSDataset, self).__init__()
        self.metadata = dataset_metadata
        self.responder_vars_idx = [84, 85, 86, 87, 88, 89, 90, 91, 92]
        self.categorical_vars_idx = [2, 3, 14, 15, 16]
        self.weight_idx = [4]
        self.time_vars_idx = [0, 1, 2]
        self.symbol_vars_idx = 3
        self.static_covariate_idx = [3]
        self.lookback_sampling = lookback_sampling
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
            + self.weight_idx
        ]
        # Adding time_idx and date_id
        self.feature_vars_idx = self.feature_vars_idx
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
        self.date_id_scaler = StandardScaler()
        self.date_id_scaler.fit(np.arange(0, 1900).reshape(-1, 1))
        self.time_idx_scaler = StandardScaler()
        self.time_idx_scaler.fit(np.arange(0, 1854468).reshape(-1, 1))
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
        symbol_hist_data: ArrayLike = self.root[f"{symbol_id}/{start_date}"].oindex[
            :: self.lookback_sampling
        ]  # type: ignore
        encoder_length = (
            (encoder_length / encoder_length) * symbol_hist_data.shape[0]
        ).type(torch.int16)
        symbol_curr_data: ArrayLike = self.root[f"{symbol_id}/{end_date}"][:][
            :decoder_length, :
        ]  # type: ignore
        # encoder_reals: ArrayLike = self._to_tensor(
        #     self.real_scaler.fit_transform(
        #         np.diff(symbol_hist_data[:, self.feature_vars_idx], n=1, prepend=0)  # type: ignore
        #     ),  # type: ignore
        #     torch.float32,
        # )
        encoder_reals: ArrayLike = self._to_tensor(
            self.real_scaler.fit_transform(
                symbol_hist_data[:, self.feature_vars_idx]  # type: ignore
            ),  # type: ignore
            torch.float32,
        )
        enc_date_id = self._to_tensor(
            self.date_id_scaler.transform(
                symbol_hist_data[:, 1].reshape(-1, 1)
            ).ravel(),
            torch.float32,
        )
        enc_time_idx = self._to_tensor(
            self.time_idx_scaler.transform(
                symbol_hist_data[:, 0].reshape(-1, 1)
            ).ravel(),
            torch.float32,
        )
        enc_wts = self._to_tensor(
            symbol_hist_data[:, self.weight_idx], torch.float32
        ).view(-1, 1)
        enc_date_time = torch.cat(
            [enc_date_id.unsqueeze(-1), enc_time_idx.unsqueeze(-1)], dim=-1
        )
        encoder_reals = torch.cat([enc_date_time, enc_wts, encoder_reals], dim=-1)
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
        # decoder_reals = self._to_tensor(
        #     self.real_scaler.transform(
        #         np.diff(symbol_curr_data[:, self.feature_vars_idx], n=1, prepend=0)  # type: ignore
        #     ),
        #     torch.float32,
        # )
        decoder_reals = self._to_tensor(
            self.real_scaler.transform(symbol_curr_data[:, self.feature_vars_idx]),
            torch.float32,
        )
        dec_date_id = self._to_tensor(
            self.date_id_scaler.transform(
                symbol_curr_data[:, 1].reshape(-1, 1)
            ).ravel(),
            torch.float32,
        )
        dec_time_idx = self._to_tensor(
            self.time_idx_scaler.transform(
                symbol_curr_data[:, 0].reshape(-1, 1)
            ).ravel(),
            torch.float32,
        )
        dec_wts = self._to_tensor(
            symbol_curr_data[:, self.weight_idx], torch.float32
        ).view(-1, 1)
        dec_date_time = torch.cat(
            [dec_date_id.unsqueeze(-1), dec_time_idx.unsqueeze(-1)], dim=-1
        )
        decoder_reals = torch.cat([dec_date_time, dec_wts, decoder_reals], dim=-1)
        decoder_categoricals = self._to_tensor(
            self.categorical_encoder.fit_transform(
                symbol_curr_data[:, self.categorical_vars_idx]  # type: ignore
            ),
            torch.int16,
        ).long()
        targets = self._to_tensor(
            symbol_curr_data[-1:, [self.target_idx, self.weight_idx]], torch.float32
        )  # type: ignore
        # encoder_length = torch.ones_like(encoder_length) * self.max_lookback
        # weights = self._to_tensor(symbol_curr_data[:, self.weight_idx], torch.float32)  # type: ignore
        # Data is returned as (num_samples, 1, features)
        return (
            {
                "encoder_length": encoder_length,
                "decoder_length": decoder_length,
                "encoder_reals": encoder_reals,
                "encoder_categoricals": encoder_categoricals,
                "encoder_targets": encoder_targets,
                "decoder_reals": decoder_reals,
                "decoder_categoricals": decoder_categoricals,
            },
            targets,
        )
