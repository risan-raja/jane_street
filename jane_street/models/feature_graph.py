from lightning.pytorch.trainer.states import RunningStage
import pytorch_lightning as ppl
from ..layers.fgraph import FeatureGraph
from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelCheckpoint,
    StochasticWeightAveraging,
    LearningRateMonitor,
)
import torch
from torchmetrics import (
    MeanSquaredError,
    R2Score,
    MeanAbsoluteError,
    NormalizedRootMeanSquaredError,
    # MeanAbsolutePercentageError,
    # RelativeSquaredError,
    # SymmetricMeanAbsolutePercentageError,
)


STAGE_STATES = {
    RunningStage.TRAINING: "train",
    RunningStage.VALIDATING: "val",
    RunningStage.TESTING: "test",
    RunningStage.PREDICTING: "predict",
    RunningStage.SANITY_CHECKING: "sanity_check",
}


class FeatureGraphModel(ppl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = FeatureGraph(config)
        self.config = config
        self.save_hyperparameters(config)
        self.mse = NormalizedRootMeanSquaredError(normalization="mean")
        self.loss = MeanSquaredError()
        self.r2 = R2Score()
        self.mae = MeanAbsoluteError()
        # self.mape = MeanAbsolutePercentageError()
        # self.rse = RelativeSquaredError()
        # self.smape = SymmetricMeanAbsolutePercentageError()
        self.metrics = {
            "mse": self.mse,
            "r2": self.r2,
            "mae": self.mae,
        }
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.example_input_array = (
            {
                "encoder_lengths": torch.tensor([10, 10, 10, 10, 10]),
                "decoder_lengths": torch.tensor([3, 3, 3, 3, 3]),
                "encoder_reals": torch.randn([5, 10, 88]),
                "encoder_categoricals": torch.ones([5, 10, 5]).long(),
                "decoder_reals": torch.randn([5, 3, 79]),
                "decoder_categoricals": torch.ones([5, 3, 5]).long(),
                "encoder_targets": torch.randn([5, 10, 9]),
            },
        )
        self.validation_step_outputs = []
        self.training_step_outputs = []

    @property
    def current_stage(self) -> str:
        """
        Available inside lightning loops.
        :return: current trainer stage. One of ["train", "val", "test", "predict", "sanity_check"]
        """
        return STAGE_STATES.get(self.trainer.state.stage, None)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_true = y.squeeze(-1)[..., 0]
        y_pred = self.model(X).squeeze(-1)
        loss = self.loss(y_pred, y_true)
        self.log(
            f"{self.current_stage}_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=X["decoder_lengths"].size(0),
            sync_dist=True,
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        _, y = batch
        y_true = y.squeeze(-1)[..., 0]
        wt = y.squeeze(-1)[..., 1]
        y_pred = outputs.squeeze(-1).clone()
        y_pred = y_pred.detach()
        metrics = {k: v(y_pred, y_true) for k, v in self.metrics.items()}
        self.log_dict(
            {f"{self.current_stage}_{k}": v for k, v in metrics.items()},
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        score, score_numerator, score_denominator = self.r2zmse_batch(
            y_pred, y_true, wt
        )
        self.log(
            f"{self.current_stage}_score_batch",
            score,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )
        # Store the numerator and denominator for the epoch level calculation
        self.training_step_outputs.append(
            [
                score_numerator,
                score_denominator,
            ]
        )

    def configure_callbacks(self):
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=False,
            mode="min",
            min_delta=1e-9,
            # stopping_threshold=1e-5,
        )
        checkpoint = ModelCheckpoint(
            dirpath="/storage/atlasAppRaja/library/atlas/model_checkpts/",
            monitor="val_loss",
            filename="{epoch}-{val_loss:.2f}-feature_graph",
            save_top_k=10,
            mode="min",
            every_n_train_steps=200,
            verbose=False,
            enable_version_counter=True,
        )
        swa = StochasticWeightAveraging(swa_lrs=1e-3, swa_epoch_start=3)
        accumulator = GradientAccumulationScheduler(scheduling={4: 2})
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [early_stopping, checkpoint, swa, accumulator, lr_monitor]

    def r2numerator(self, y_pred, y_true, wt):
        return (((y_pred - y_true) ** 2) * wt).sum()

    def r2denominator(self, y_true, wt):
        return ((y_true**2) * wt).sum() + 1e-9

    def r2zmse_batch(self, y_pred, y_true, wt):
        num = self.r2numerator(y_pred, y_true, wt)
        den = self.r2denominator(y_true, wt)
        result = 1 - (num / den)
        return result, num, den

    def score_output(self, stored_outputs):
        return 1 - torch.sum(
            torch.tensor(
                [val[0] for val in stored_outputs],
                device=stored_outputs[0][0].device,
            ),
        ) / torch.sum(
            torch.tensor(
                [val[1] for val in stored_outputs],
                device=stored_outputs[0][0].device,
            )
        )

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_true = y.squeeze(-1)[..., 0]
        y_pred = self.model(X).squeeze(-1)
        loss = self.loss(y_pred, y_true)
        self.log(
            f"{self.current_stage}_loss",
            loss,
            # prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=X["decoder_lengths"].size(0),
            sync_dist=True,
        )
        return loss

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        _, y = batch
        y_true = y.squeeze(-1)[..., 0]
        wt = y.squeeze(-1)[..., 1]
        y_pred = outputs.squeeze(-1).clone()
        y_pred = y_pred.detach()
        metrics = {k: v(y_pred, y_true) for k, v in self.metrics.items()}
        self.log_dict(
            {f"{self.current_stage}_{k}": v for k, v in metrics.items()},
            on_epoch=True,
            sync_dist=True,
        )
        score, score_numerator, score_denominator = self.r2zmse_batch(
            y_pred, y_true, wt
        )
        self.log(
            f"{self.current_stage}_score_batch",
            score,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )
        # Store the numerator and denominator for the epoch level calculation
        self.validation_step_outputs.append(
            [
                score_numerator,
                score_denominator,
            ]
        )

    def on_validation_epoch_end(self):
        zrmse = self.score_output(self.validation_step_outputs)
        self.log(f"{self.current_stage}_score", zrmse, sync_dist=True, prog_bar=True)
        self.log("hp_metric", zrmse, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        zrmse = self.score_output(self.training_step_outputs)
        self.log(f"{self.current_stage}_zrmse", zrmse, sync_dist=True, prog_bar=True)
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 6,
            },
        }

    def forward(self, X):
        return self.model(X)
