from pytorch_lightning.trainer.states import RunningStage
from typing import Any, Dict
from omegaconf import DictConfig
import pytorch_lightning as ppl
import numpy as np
import matplotlib.pyplot as plt
import torch
from ..layers.tft import TFT

# from ..metrics.quantile import WeightedQuantileLoss
# from ..metrics.r2score import WeightedZeroMeanR2Score
from ..metrics.point import SMAPE, MSSE, MAE, MSE, nMSE
from ..utils.outputs import detach, create_mask, integer_histogram, masked_op
from ..utils.pad import padded_stack

STAGE_STATES = {
    RunningStage.TRAINING: "train",
    RunningStage.VALIDATING: "val",
    RunningStage.TESTING: "test",
    RunningStage.PREDICTING: "predict",
    RunningStage.SANITY_CHECKING: "sanity_check",
}


class TemporalFT(ppl.LightningModule):
    def __init__(self, master_conf: DictConfig):
        super().__init__()
        self.save_hyperparameters(master_conf)
        self.config = master_conf
        self.model = TFT(master_conf)
        # self.wq_loss = WeightedQuantileLoss(master_conf.quantiles)
        self.r_loss = MSSE()
        self.loss = MSE()
        # self.train_weighted_r2 = WeightedZeroMeanR2Score()
        # self.val_weighted_r2 = WeightedZeroMeanR2Score()
        # self.wt_loss = ZRMSS(master_conf.quantiles)
        self.logging_metrics = [SMAPE(), MAE(), self.r_loss, nMSE()]
        # self.logging_metrics = [SMAPE()]
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.learning_rate = master_conf.learning_rate
        self.max_encoder_length = master_conf.max_encoder_length

    @property
    def static_categoricals(self):
        print(list(self.config.static_categoricals))
        return list(self.config.static_categoricals)

    @property
    def encoder_variables(self):
        return (
            list(self.config.x_categoricals)
            + list(self.config.real_variables)
            + list(self.config.responder_variables)
        )

    @property
    def decoder_variables(self):
        return list(self.config.x_categoricals) + list(self.config.real_variables)

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.weight_decay,
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

    def log(self, *args, **kwargs):
        """See :meth:`lightning.pytorch.core.lightning.LightningModule.log`."""
        # never log for prediction
        if not self.predicting:
            super().log(*args, **kwargs)

    def to_prediction(self, out: Dict[str, Any], use_metric: bool = True, **kwargs):
        """
        Convert output to prediction using the loss metric.

        Args:
            out (Dict[str, Any]): output of network where "prediction" has been
                transformed with :py:meth:`~transform_output`
            use_metric (bool): if to use metric to convert for conversion, if False,
                simply take the average over ``out["prediction"]``
            **kwargs: arguments to metric ``to_quantiles`` method

        Returns:
            torch.Tensor: predictions of shape batch_size x timesteps
        """
        return self.loss.to_prediction(out["prediction"])

    @property
    def predicting(self) -> bool:
        return self.current_stage is None or self.current_stage == "predict"

    @property
    def current_stage(self) -> str:
        """
        Available inside lightning loops.
        :return: current trainer stage. One of ["train", "val", "test", "predict", "sanity_check"]
        """
        return STAGE_STATES.get(self.trainer.state.stage, None)

    @property
    def log_interval(self) -> float:
        """
        Log interval depending if training or validating
        """
        if self.training:
            return self.config.log_interval
        elif self.predicting:
            return -1
        else:
            return self.config.log_val_interval

    def size(self) -> int:
        """
        get number of parameters in model
        """
        return sum(p.numel() for p in self.parameters())

    def training_step(self, batch, batch_idx):
        """
        Train on batch.
        """
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        self.training_step_outputs.append(log)
        if self.global_step % self.log_interval == 0:
            self.log_metrics(y, out, batch_idx)
        # if self.global_step % self.log_interval*5 == 0:
        #     # self.log_interpretation(self.training_step_output
        #     self.training_step_outputs.clear()
        # self.on_train_epoch_end()
        # self.train_weighted_r2.update(out["prediction"].detach().squeeze(), y.detach().squeeze()[...,0], y.detach().squeeze()[...,1])
        return log

    def on_train_epoch_end(self):
        self.on_epoch_end(self.training_step_outputs)
        # train_r2 = self.train_weighted_r2.compute()
        self.training_step_outputs.clear()
        # self.log("hp/train_weighted_r2", train_r2, sync_dist=True)
        # self.train_weighted_r2.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        # self.val_weighted_r2.update(out["prediction"].detach().squeeze(), y.squeeze()[...,0], y.squeeze()[...,1])
        self.validation_step_outputs.append(log)
        if self.global_step % self.log_interval == 0:
            self.log_metrics(y, out, batch_idx)
        # if self.global_step % self.log_interval*5 == 0:
        #     self.validation_step_outputs.clear()
        # self.on_validation_epoch_end()
        # self.on_train_epoch_end()
        return log

    def on_epoch_end(self, outputs):
        """
        run at epoch end for training or validation
        """
        return None
        # if self.log_interval > 0 and not self.training:
        # self.log_interpretation(outputs)
        # self.log_metrics()

    def on_validation_epoch_end(self):
        self.on_epoch_end(self.validation_step_outputs)
        # val_r2 = self.val_weighted_r2.compute()
        # self.log("hp/val_weighted_r2", val_r2, sync_dist=True)
        # self.val_weighted_r2.reset()
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.testing_step_outputs.append(log)
        return log

    def on_test_epoch_end(self):
        self.on_epoch_end(self.testing_step_outputs)
        self.testing_step_outputs.clear()

    def step(
        self, x: dict[str, torch.Tensor], y: torch.Tensor, batch_idx: int
    ) -> tuple[dict, dict]:
        """
        Train on batch.
        """
        # pack y sequence if different encoder lengths exist
        out = self(x)
        prediction = out["prediction"]
        loss = self.loss(prediction, y)
        self.log(
            f"{self.current_stage}_loss",
            loss,
            # prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(x["decoder_lengths"]),
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            f"hp/{self.current_stage}_loss",
            loss,
            # prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(x["decoder_lengths"]),
            sync_dist=True,
        )
        # self.log(
        #     f"{self.current_stage}_max_decoder_length",
        #     int(x["decoder_lengths"].max().item()),
        #     on_step=True,
        #     on_epoch=True,
        #     batch_size=len(x["decoder_lengths"]),
        #     sync_dist=True,
        # )
        # self.log
        # self.log_metrics()
        log = {"loss": loss, "n_samples": x["decoder_lengths"].size(0)}
        return log, out

    def log_metrics(self, target, prediction, batch_idx):
        # print("Triggered")
        y_hat_point = prediction["prediction"].detach()
        for metric in self.logging_metrics:
            loss_value = metric(y_hat_point, target)
            self.log(
                f"{self.current_stage}_{metric.name}",
                loss_value,
                on_step=self.training,
                on_epoch=True,
                sync_dist=True,
            )

    def create_log(self, x, y, out, batch_idx, **kwargs):
        self.log_metrics(y, out, batch_idx)
        # log = super().create_log(x, y, out, batch_idx, **kwargs)
        log = {}
        # if self.log_interval > 0:
        #     log["interpretation"] = self._log_interpretation(out)
        return log

    def _log_interpretation(self, out):
        # calculate interpretations etc for latter logging
        if self.config.skip_interpretation:
            return {}
        interpretation = self.interpret_output(
            detach(out),
            reduction="sum",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        return interpretation

    def interpret_output(
        self,
        out: Dict[str, torch.Tensor],
        reduction: str = "none",
        attention_prediction_horizon: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        interpret output of model

        Args:
            out: output as produced by ``forward()``
            reduction: "none" for no averaging over batches, "sum" for summing attentions, "mean" for
                normalizing by encode lengths
            attention_prediction_horizon: which prediction horizon to use for attention

        Returns:
            interpretations that can be plotted with ``plot_interpretation()``
        """
        # take attention and concatenate if a list to proper attention object
        if self.config.skip_interpretation:
            return {}
        max_encoder_length = int(out["encoder_lengths"].max())
        encoder_attention = out["attention"][..., :max_encoder_length]
        decoder_attention = out["attention"][..., max_encoder_length:]
        batch_size = len(decoder_attention)
        if isinstance(decoder_attention, (list, tuple)):
            # start with decoder attention
            # assume issue is in last dimension, we need to find max
            max_last_dimension = max(x.size(-1) for x in out["decoder_attention"])
            first_elm = decoder_attention[0]
            # create new attention tensor into which we will scatter
            decoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], max_last_dimension),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            for idx, x in enumerate(decoder_attention):
                decoder_length = out["decoder_lengths"][idx]
                decoder_attention[idx, :, :, :decoder_length] = x[..., :decoder_length]
        else:
            decoder_attention = decoder_attention.clone()
            decoder_mask = create_mask(
                decoder_attention.size(1), out["decoder_lengths"]
            )
            decoder_attention[
                decoder_mask[..., None, None].expand_as(decoder_attention)
            ] = float("nan")

        if isinstance(encoder_attention, (list, tuple)):
            # same game for encoder attention
            # create new attention tensor into which we will scatter
            first_elm = encoder_attention[0]
            encoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], self.max_encoder_length),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            for idx, x in enumerate(encoder_attention):
                encoder_length = encoder_attention[idx]
                encoder_attention[
                    idx, :, :, self.max_encoder_length - encoder_length :
                ] = x[..., :encoder_length]
        else:
            # roll encoder attention (so start last encoder value is on the right)
            encoder_attention = encoder_attention.clone()
            shifts = encoder_attention.size(3) - out["encoder_lengths"]
            new_index = (
                torch.arange(
                    encoder_attention.size(3), device=encoder_attention.device
                )[None, None, None].expand_as(encoder_attention)
                - shifts[:, None, None, None]
            ) % encoder_attention.size(3)
            encoder_attention = torch.gather(encoder_attention, dim=3, index=new_index)
            # expand encoder_attention to full size
            if encoder_attention.size(-1) < self.max_encoder_length:
                encoder_attention = torch.concat(
                    [
                        torch.full(
                            (
                                *encoder_attention.shape[:-1],
                                self.max_encoder_length - out["encoder_lengths"].max(),
                            ),
                            float("nan"),
                            dtype=encoder_attention.dtype,
                            device=encoder_attention.device,
                        ),
                        encoder_attention,
                    ],
                    dim=-1,
                )

        # combine attention vector
        attention = torch.concat([encoder_attention, decoder_attention], dim=-1)
        attention[attention < 1e-5] = float("nan")

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(
            out["encoder_lengths"], min=0, max=self.max_encoder_length
        )
        decoder_length_histogram = integer_histogram(
            out["decoder_lengths"], min=1, max=out["decoder_variables"].size(1)
        )

        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = out["encoder_variables"].squeeze(-2).clone()
        encode_mask = create_mask(encoder_variables.size(1), out["encoder_lengths"])
        encoder_variables = encoder_variables.masked_fill(
            encode_mask.unsqueeze(-1), 0.0
        ).sum(dim=1)
        encoder_variables /= (
            out["encoder_lengths"]
            .where(out["encoder_lengths"] > 0, torch.ones_like(out["encoder_lengths"]))
            .unsqueeze(-1)
        )

        decoder_variables = out["decoder_variables"].squeeze(-2).clone()
        decode_mask = create_mask(decoder_variables.size(1), out["decoder_lengths"])
        decoder_variables = decoder_variables.masked_fill(
            decode_mask.unsqueeze(-1), 0.0
        ).sum(dim=1)
        decoder_variables /= out["decoder_lengths"].unsqueeze(-1)

        # static variables need no masking
        static_variables = out["static_variables"].squeeze(1)
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention and attention on observed timesteps
        attention = masked_op(
            attention[
                :,
                attention_prediction_horizon,
                :,
                : self.max_encoder_length + attention_prediction_horizon,
            ],
            op="mean",
            dim=1,
        )

        if reduction != "none":  # if to average over batches
            static_variables = static_variables.sum(dim=0)
            encoder_variables = encoder_variables.sum(dim=0)
            decoder_variables = decoder_variables.sum(dim=0)

            attention = masked_op(attention, dim=0, op=reduction)
        else:
            attention = attention / masked_op(attention, dim=1, op="sum").unsqueeze(
                -1
            )  # renormalize

        interpretation = dict(
            attention=attention.masked_fill(torch.isnan(attention), 0.0),
            static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        return interpretation

    def log_interpretation(self, outputs):
        """
        Log interpretation metrics to tensorboard.
        """
        if self.config.skip_interpretation:
            return
        # extract interpretations
        interpretation = {
            # use padded_stack because decoder length histogram can be of different length
            name: padded_stack(
                [x["interpretation"][name].detach() for x in outputs],
                side="right",
                value=0,
            ).sum(0)
            for name in outputs[0]["interpretation"].keys()
        }
        # normalize attention with length histogram squared to account for: 1. zeros in attention and
        # 2. higher attention due to less values
        attention_occurances = (
            interpretation["encoder_length_histogram"][1:].flip(0).float().cumsum(0)
        )
        attention_occurances = attention_occurances / attention_occurances.max()
        attention_occurances = torch.cat(
            [
                attention_occurances,
                torch.ones(
                    interpretation["attention"].size(0) - attention_occurances.size(0),
                    dtype=attention_occurances.dtype,
                    device=attention_occurances.device,
                ),
            ],
            dim=0,
        )
        interpretation["attention"] = interpretation[
            "attention"
        ] / attention_occurances.pow(2).clamp(1.0)
        interpretation["attention"] = (
            interpretation["attention"] / interpretation["attention"].sum()
        )
        # torch.save(interpretation, 'interpretation.pt')
        figs = self.plot_interpretation(interpretation)  # make interpretation figures
        label = self.current_stage
        # log to tensorboard
        for name, fig in figs.items():
            self.logger.experiment.add_figure(
                f"{label.capitalize()} {name} importance",
                fig,
                global_step=self.global_step,
            )

        # log lengths of encoder/decoder
        for type in ["encoder", "decoder"]:
            fig, ax = plt.subplots()
            lengths = (
                padded_stack(
                    [
                        out["interpretation"][f"{type}_length_histogram"]
                        for out in outputs
                    ]
                )
                .sum(0)
                .detach()
                .cpu()
            )
            if type == "decoder":
                start = 1
            else:
                start = 0
            ax.plot(torch.arange(start, start + len(lengths)), lengths)
            ax.set_xlabel(f"{type.capitalize()} length")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"{type.capitalize()} length distribution in {label} epoch")

            self.logger.experiment.add_figure(
                f"{label.capitalize()} {type} length distribution",
                fig,
                global_step=self.global_step,
            )
        # self.log('interpretation', interpretation)

    def plot_interpretation(self, interpretation: Dict[str, torch.Tensor]):
        """
        Make figures that interpret model.

        * Attention
        * Variable selection weights / importances

        Args:
            interpretation: as obtained from ``interpret_output()``

        Returns:
            dictionary of matplotlib figures
        """

        figs = {}

        # attention
        fig, ax = plt.subplots()
        attention = interpretation["attention"].detach().cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        ax.plot(
            np.arange(
                -self.max_encoder_length, attention.size(0) - self.max_encoder_length
            ),
            attention,
        )
        ax.set_xlabel("Time index")
        ax.set_ylabel("Attention")
        ax.set_title("Attention")
        figs["attention"] = fig

        # variable selection
        def make_selection_plot(title, values, labels):
            fig, ax = plt.subplots(figsize=(7, len(values) * 0.25 + 2))
            order = np.argsort(values)
            values = values / values.sum(-1).unsqueeze(-1)
            ax.barh(
                np.arange(len(values)),
                values[order].ravel() * 100,
                tick_label=np.asarray(labels)[order].ravel(),
            )
            ax.set_title(title)
            ax.set_xlabel("Importance in %")
            plt.tight_layout()
            return fig

        figs["static_variables"] = make_selection_plot(
            "Static variables importance",
            interpretation["static_variables"].detach().cpu(),
            self.static_categoricals,
        )
        figs["encoder_variables"] = make_selection_plot(
            "Encoder variables importance",
            interpretation["encoder_variables"].detach().cpu(),
            self.encoder_variables,
        )
        figs["decoder_variables"] = make_selection_plot(
            "Decoder variables importance",
            interpretation["decoder_variables"].detach().cpu(),
            self.decoder_variables,
        )

        return figs
