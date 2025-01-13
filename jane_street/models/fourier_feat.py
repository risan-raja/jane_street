from pytorch_lightning.trainer.states import RunningStage
from typing import Optional
import pytorch_lightning as ppl
from ..layers.fourier import FourierFastPred, FourierIndexPred
from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelCheckpoint,
    # StochasticWeightAveraging,
    LearningRateMonitor,
)
import math
import torch
from torchmetrics import (
    # MeanAbsolutePercentageError,
    MeanSquaredError,
    R2Score,
    MeanAbsoluteError,
    NormalizedRootMeanSquaredError,
    RelativeSquaredError,
    MetricCollection,
    # SymmetricMeanAbsolutePercentageError,
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


class FourierModel(ppl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = FourierFastPred(config)
        self.config = config
        self.save_hyperparameters(config)
        self.nmse = NormalizedRootMeanSquaredError(normalization="mean")
        self.mse = MeanSquaredError()
        self.r2 = R2Score()
        self.mae = MeanAbsoluteError()
        # self.mape = MeanAbsolutePercentageError()
        self.loss = RelativeSquaredError()
        # self.smape = SymmetricMeanAbsolutePercentageError()
        metrics = MetricCollection(
            {
                "r2": self.r2,
                "mse": self.mse,
                "mae": self.mae,
                "nmse": self.nmse,
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
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
        self.objective = "min" if config.objective == "min" else "max"
        if isinstance(self.loss, R2Score):
            self.objective = "max"
        self.bootstrap_metric = "val_score"

    @property
    def current_stage(self) -> str:
        """
        Available inside lightning loops.
        :return: current trainer stage. One of ["train", "val", "test", "predict", "sanity_check"]
        """
        return STAGE_STATES.get(self.trainer.state.stage, None)

    def sep_wt_target(self, y):
        return y.squeeze(-1)[..., 0], y.squeeze(-1)[..., 1]

    def to_prediction(self, y_pred: torch.Tensor):
        if y_pred.ndim == 3:
            B, _, D = y_pred.size()
            return y_pred.reshape((B, D))
        if y_pred.ndim == 2:
            return y_pred

    def get_optimizer(
        self,
        lr: Optional[float] = 1e-3,
        lambda_lr: Optional[float] = 0.1,
        weight_decay: Optional[float] = 1e-2,
        maximize: Optional[bool] = False,
    ) -> torch.optim.Optimizer:
        group1 = []  # lambda
        group2 = []  # no decay
        group3 = []  # decay
        no_decay_list = (
            "bias",
            "norm",
        )
        for param_name, param in self.model.named_parameters():
            if "_lambda" in param_name:
                group1.append(param)
            elif any([mod in param_name for mod in no_decay_list]):
                group2.append(param)
            else:
                group3.append(param)
        optimizer = torch.optim.Adam(
            [
                {
                    "params": group1,
                    "weight_decay": 0,
                    "lr": lambda_lr,
                    "scheduler": "cosine_annealing",
                },
                {
                    "params": group2,
                    "weight_decay": 0,
                    "scheduler": "cosine_annealing_with_linear_warmup",
                },
                {"params": group3, "scheduler": "cosine_annealing_with_linear_warmup"},
            ],
            lr=lr,
            weight_decay=weight_decay,
            maximize=self.objective == "max",
        )
        return optimizer

    def log_loss(self, loss, batch_size):
        self.log(
            f"{self.current_stage}_loss",
            loss if loss.size() == 1 else loss.mean(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def log_metrics(self, y_pred, y_true):
        if self.current_stage == "train":
            metrics = self.train_metrics(y_pred, y_true)
        else:
            metrics = self.val_metrics(y_pred, y_true)

        self.log_dict(
            {k: v if v.size() == 1 else v.mean() for k, v in metrics.items()},
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_true, _ = self.sep_wt_target(y)
        y_pred = self.to_prediction(self.model(X))
        loss = self.loss(y_pred, y_true)
        self.log_loss(loss, X["decoder_lengths"].size(0))
        return {"loss": loss, "prediction": y_pred}

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        _, y = batch
        y_true, wt = self.sep_wt_target(y)
        y_pred = outputs["prediction"].clone()
        y_pred = y_pred.detach()
        self.log_metrics(y_pred, y_true)
        score_numerator, score_denominator = self.log_zrmse_batch(y_pred, y_true, wt)
        if batch_idx == 0 and self.current_epoch == 0:
            self.log(
                self.bootstrap_metric,
                1
                - (
                    score_numerator / score_denominator
                ).mean(),  # Safety for multi-output
                on_step=True,
                sync_dist=True,
            )
        # Store the numerator and denominator for the epoch level calculation
        self.training_step_outputs.append(
            [
                score_numerator,
                score_denominator,
            ]
        )

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_true, _ = self.sep_wt_target(y)
        y_pred = self.to_prediction(self.model(X))
        loss = self.loss(y_pred, y_true)
        self.log_loss(loss, X["decoder_lengths"].size(0))
        return {"loss": loss, "prediction": y_pred}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        _, y = batch
        y_true, wt = self.sep_wt_target(y)
        y_pred = outputs["prediction"].clone()
        y_pred = y_pred.detach()
        self.log_metrics(y_pred, y_true)
        score_numerator, score_denominator = self.log_zrmse_batch(y_pred, y_true, wt)
        # Store the numerator and denominator for the epoch level calculation
        self.validation_step_outputs.append(
            [
                score_numerator,
                score_denominator,
            ]
        )

    def on_validation_epoch_end(self):
        self.log_zrmse_epoch(self.validation_step_outputs)
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        self.log_zrmse_epoch(self.training_step_outputs)
        self.training_step_outputs.clear()

    def log_zrmse_batch(self, y_pred, y_true, wt):
        score, score_numerator, score_denominator = self.r2zmse_batch(
            y_pred, y_true, wt
        )
        self.log(
            f"{self.current_stage}_score_batch",
            score,
            on_step=True,
            on_epoch=False,
            # rank_zero_only=True,
            sync_dist=True,
        )
        return score_numerator, score_denominator

    def log_zrmse_epoch(self, outputs):
        zrmse = self.score_output(outputs)
        self.log(f"{self.current_stage}_score", zrmse, sync_dist=True, prog_bar=True)
        if self.current_stage == "val":
            self.log("hp_metric", zrmse, on_epoch=True, sync_dist=True)

    def configure_callbacks(self):
        early_stopping = EarlyStopping(
            monitor="val_score",
            patience=10,
            verbose=False,
            mode="max",
            min_delta=1e-5,
            # stopping_threshold=1e-5,
        )
        checkpoint = ModelCheckpoint(
            dirpath="/storage/atlasAppRaja/library/atlas/model_checkpts/",
            monitor="val_score",
            filename="{epoch}-{val_score:.2f}-fourier-fastpred",
            save_top_k=10,
            mode="max",
            every_n_train_steps=200,
            verbose=False,
            enable_version_counter=True,
        )
        backup_ckpt = ModelCheckpoint(
            dirpath="/storage/atlasAppRaja/library/atlas/model_checkpts/",
            monitor="val_score",
            filename="{epoch}-{val_score:.2f}-fourier-fastpred-backup",
            # save_top_k=10,
            mode="max",
            # every_n_train_steps=200,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            verbose=False,
            enable_version_counter=True,
        )
        # swa = StochasticWeightAveraging(swa_lrs=1e-3, swa_epoch_start=2)
        accumulator = GradientAccumulationScheduler(scheduling={5: 4})
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [early_stopping, checkpoint, accumulator, lr_monitor, backup_ckpt]

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
        return 1 - (
            torch.sum(
                torch.tensor(
                    [val[0] for val in stored_outputs],
                    device=stored_outputs[0][0].device,
                ),
            )
            / torch.sum(
                torch.tensor(
                    [val[1] for val in stored_outputs],
                    device=stored_outputs[0][0].device,
                )
            )
        )

    def configure_optimizers(self):
        optimizer = self.get_optimizer(
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            maximize=self.objective == "max",
        )
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", patience=5, factor=0.8
        # )
        lr_scheduler = self.get_scheduler(optimizer, 50, 5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def forward(self, X):
        return self.model(X)

    def get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        warmup_epochs: int,
        eta_min: Optional[float] = 0.00,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        def identity_fn(T_cur):
            return 1

        def fn_1(T_cur):
            return (
                eta_min
                + 0.5
                * (eta_max - eta_min)
                * (
                    1.0
                    + math.cos(
                        (T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi
                    )
                )
            ) / lr

        # https://blog.csdn.net/qq_36560894/article/details/114004799
        def fn_2(T_cur):
            return (
                T_cur / warmup_epochs
                if T_cur < warmup_epochs
                else (
                    eta_min
                    + 0.5
                    * (eta_max - eta_min)
                    * (
                        1.0
                        + math.cos(
                            (T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi
                        )
                    )
                )
                / lr
            )

        scheduler_fns = []
        for param_group in optimizer.param_groups:
            scheduler = param_group["scheduler"]
            if scheduler == "none":
                fn = identity_fn
            elif scheduler == "cosine_annealing":
                lr = eta_max = param_group["lr"]
                fn = fn_1
            elif scheduler == "cosine_annealing_with_linear_warmup":
                lr = eta_max = param_group["lr"]
                fn = fn_2
            else:
                raise ValueError(f"No such scheduler, {scheduler}")
            scheduler_fns.append(fn)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=scheduler_fns
        )
        return scheduler


class FourierIndexModel(FourierModel):
    def __init__(self, config):
        super().__init__(config)
        self.loss = R2Score()
        self.rse = RelativeSquaredError()
        metrics = MetricCollection(
            {
                "rse": self.rse,
                "mse": self.mse,
                "mae": self.mae,
                "nmse": self.nmse,
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.model = FourierIndexPred(config)
        self.objective = "max"
