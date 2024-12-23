from typing import Any
from dataclasses import dataclass, field
from tempfile import NamedTemporaryFile
import zarr
from numpy.typing import NDArray
from collections import OrderedDict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pickle
import json
import polars as pl


@dataclass
class JSDatasetMeta:
    """
    A class to manage metadata for a temporal dataset stored in Zarr format.
    Symbols traded on a particular day that do not have any previous date data are removed from the dataset.
    This is to ensure that the dataset is consistent and does not contain any missing values.

    Attributes:
    -----------
    date_data_dist_path : str
        Path to the JSON file historical sequence length and the number of symbols for each date.
    date_symbol_path : str
        Path to the JSON file containing symbols traded for that particular day.
    symbols_prev_date_path : str
        Path to the JSON file containing the previous date for the symbols traded on that day.
    zaar_root_path : str
        Path to the root of the Zarr dataset.
    self.synchronizer : zarr.ProcessSynchronizer
        Synchronizer for the Zarr dataset.
    self.root : zarr.hierarchy.Group
        Root group of the Zarr dataset.
    self.date_data_dist : dict[int, dict[tuple[int, int], int]]
        (Historical sequence length dimensionality) : the number of symbols for each date.
    self.date_symbol : dict[int, list[int]]
        Symbols traded on that particular day.
    self.symbols_prev_date : dict[int, list[int]]
        Previous date for the symbols traded on that day.

    Methods:
    --------
    load_json(path):
        Loads a JSON file from the given path and returns its content.

    __post_init__():
        Initializes the Zarr group and loads the JSON metadata files.
    """

    dataloc = "/storage/atlasAppRaja/library/kaggle/data"
    date_data_dist_path: str = field(default=f"{dataloc}/date_data_dist_strict.json")
    date_symbol_path: str = field(default=f"{dataloc}/date_symbol_strict.json")
    symbols_prev_date_path: str = field(
        default=f"{dataloc}/prev_date_symbols_strict.json"
    )
    categories_path: str = field(default=f"{dataloc}/categories.json")
    zaar_root_path: str = field(default=f"{dataloc}/train_symbol_fast.zarr")
    feature_scaler_path: str = field(default=f"{dataloc}/feature_scaler.pkl")
    index_path: str = field(default=f"{dataloc}/symbdf/symbdf_cat_train.parquet")
    # synchronizer = zarr.ProcessSynchronizer('/var/run/zarr.lock')
    # sampler_index_path: str = field(default="../data/symbdf/symbdf_cat_train.parquet")

    @staticmethod
    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __post_init__(self):
        self.store = zarr.DirectoryStore(self.zaar_root_path)
        self.synchronizer: zarr.ProcessSynchronizer = zarr.ProcessSynchronizer(
            NamedTemporaryFile().name
        )
        self.root: zarr.Group = zarr.group(
            store=self.store, synchronizer=self.synchronizer
        )
        self.date_data_dist: dict[int, dict[tuple[int, int], int]] = self.load_json(
            self.date_data_dist_path
        )
        self.date_symbol: dict[int, list[int]] = self.load_json(self.date_symbol_path)
        self.symbols_prev_date: dict[int, dict[int, list[int]]] = self.load_json(
            self.symbols_prev_date_path
        )
        self.categories: dict[str, list[str]] = self.load_json(self.categories_path)
        self.idx_date: list[int] = list(self.date_data_dist.keys())
        self.feature_scaler: StandardScaler = self.load_pickle(self.feature_scaler_path)
        self.index: pl.DataFrame = pl.read_parquet(self.index_path)
        # self.sampler_index: pl.DataFrame = pl.read_parquet(self.sampler_index_path)
        self.sampler_index = self.index.with_row_index("idx")


if __name__ == "__main__":
    meta = JSDatasetMeta()
    print(len(meta.sampler_index))
