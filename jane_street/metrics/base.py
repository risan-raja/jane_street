import inspect
from typing import Any, Callable, List, Optional, Dict
import warnings

from sklearn.base import BaseEstimator
import torch
from torch.nn.utils import rnn
from torchmetrics import Metric as LightningMetric
from ..utils.pad import unpack_sequence, unsqueeze_like


class Metric(LightningMetric):
    """
    Base metric class that has basic functions that can handle predicting quantiles and operate in log space.
    See the `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/latest/metrics.html>`_
    for details of how to implement a new metric

    Other metrics should inherit from this base class
    """

    full_state_update = False
    higher_is_better = False
    is_differentiable = True

    def __init__(
        self,
        name: str = None,
        quantiles: List[float] = None,
        reduction="mean",
        **kwargs,
    ):
        """
        Initialize metric

        Args:
            name (str): metric name. Defaults to class name.
            quantiles (List[float], optional): quantiles for probability range. Defaults to None.
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
        """
        self.quantiles = quantiles
        self.reduction = reduction
        if name is None:
            name = self.__class__.__name__
        self.name = name
        super().__init__(**kwargs)

    def update(self, y_pred: torch.Tensor, y_actual: torch.Tensor):
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        """
        Abstract method that calcualtes metric

        Should be overriden in derived classes

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        raise NotImplementedError()

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        """
        Rescale normalized parameters into the scale required for the output.

        Args:
            parameters (torch.Tensor): normalized parameters (indexed by last dimension)
            target_scale (torch.Tensor): scale of parameters (n_batch_samples x (center, scale))
            encoder (BaseEstimator): original encoder that normalized the target in the first place

        Returns:
            torch.Tensor: parameters in real/not normalized space
        """
        return encoder(dict(prediction=parameters, target_scale=target_scale))

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            if self.quantiles is None:
                assert (
                    y_pred.size(-1) == 1
                ), "Prediction should only have one extra dimension"
                y_pred = y_pred[..., 0]
            else:
                y_pred = y_pred.mean(-1)
        return y_pred

    def to_quantiles(
        self, y_pred: torch.Tensor, quantiles: List[float] = None
    ) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network
            quantiles (List[float], optional): quantiles for probability range. Defaults to quantiles as
                as defined in the class initialization.

        Returns:
            torch.Tensor: prediction quantiles
        """
        if quantiles is None:
            quantiles = self.quantiles

        if y_pred.ndim == 2:
            return y_pred.unsqueeze(-1)
        elif y_pred.ndim == 3:
            if y_pred.size(2) > 1:  # single dimension means all quantiles are the same
                assert quantiles is not None, "quantiles are not defined"
                y_pred = torch.quantile(
                    y_pred, torch.tensor(quantiles, device=y_pred.device), dim=2
                ).permute(1, 2, 0)
            return y_pred
        else:
            raise ValueError(
                f"prediction has 1 or more than 3 dimensions: {y_pred.ndim}"
            )

    def __add__(self, metric: LightningMetric):
        composite_metric = CompositeMetric(metrics=[self])
        new_metric = composite_metric + metric
        return new_metric

    def __mul__(self, multiplier: float):
        new_metric = CompositeMetric(metrics=[self], weights=[multiplier])
        return new_metric

    def extra_repr(self) -> str:
        forbidden_attributes = ["name", "reduction"]
        attributes = list(inspect.signature(self.__class__).parameters.keys())
        return ", ".join(
            [
                f"{name}={repr(getattr(self, name))}"
                for name in attributes
                if hasattr(self, name) and name not in forbidden_attributes
            ]
        )

    __rmul__ = __mul__


class CompositeMetric(LightningMetric):
    """
    Metric that combines multiple metrics.

    Metric does not have to be called explicitly but is automatically created when adding and multiplying metrics
    with each other.

    Example:

        .. code-block:: python

            composite_metric = SMAPE() + 0.4 * MAE()
    """

    full_state_update = False
    higher_is_better = False
    is_differentiable = True

    def __init__(
        self,
        metrics: Optional[List[LightningMetric]] = None,
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            metrics (List[LightningMetric], optional): list of metrics to combine. Defaults to None.
            weights (List[float], optional): list of weights / multipliers for weights. Defaults to 1.0 for all metrics.
        """
        self.metrics = metrics
        self.weights = weights

        if metrics is None:
            metrics = []
        if weights is None:
            weights = [1.0 for _ in metrics]
        assert len(weights) == len(
            metrics
        ), "Number of weights has to match number of metrics"

        self._metrics = list(metrics)
        self._weights = list(weights)

        super().__init__()

    def __repr__(self):
        name = " + ".join(
            [
                f"{w:.3g} * {repr(m)}" if w != 1.0 else repr(m)
                for w, m in zip(self._weights, self._metrics)
            ]
        )
        return name

    def update(self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs):
        """
        Update composite metric

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        for metric in self._metrics:
            try:
                metric.update(y_pred, y_actual, **kwargs)
            except TypeError:
                metric.update(y_pred, y_actual)

    def compute(self) -> torch.Tensor:
        """
        Get metric

        Returns:
            torch.Tensor: metric
        """
        results = []
        for weight, metric in zip(self._weights, self._metrics):
            results.append(metric.compute() * weight)

        if len(results) == 1:
            results = results[0]
        else:
            results = torch.stack(results, dim=0).sum(0)
        return results

    @torch.jit.unused
    def forward(self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs):
        """
        Calculate composite metric

        Args:
            y_pred: network output
            y_actual: actual values
            **kwargs: arguments to update function

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        results = []
        for weight, metric in zip(self._weights, self._metrics):
            try:
                results.append(metric(y_pred, y_actual, **kwargs) * weight)
            except TypeError:
                results.append(metric(y_pred, y_actual) * weight)

        if len(results) == 1:
            results = results[0]
        else:
            results = torch.stack(results, dim=0).sum(0)
        return results

    def _wrap_compute(self, compute: Callable) -> Callable:
        return compute

    def _sync_dist(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
    ) -> None:
        # No syncing required here. syncing will be done in metrics
        pass

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()

    def persistent(self, mode: bool = False) -> None:
        for metric in self._metrics:
            metric.persistent(mode=mode)

    def to_prediction(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Will use first metric in ``metrics`` attribute to calculate result.

        Args:
            y_pred: prediction output of network
            **kwargs: parameters to first metric `to_prediction` method

        Returns:
            torch.Tensor: point prediction
        """
        return self._metrics[0].to_prediction(y_pred, **kwargs)

    def to_quantiles(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Will use first metric in ``metrics`` attribute to calculate result.

        Args:
            y_pred: prediction output of network
            **kwargs: parameters to first metric's ``to_quantiles()`` method

        Returns:
            torch.Tensor: prediction quantiles
        """
        return self._metrics[0].to_quantiles(y_pred, **kwargs)

    def __add__(self, metric: LightningMetric):
        if isinstance(metric, self.__class__):
            self._metrics.extend(metric._metrics)
            self._weights.extend(metric._weights)
        else:
            self._metrics.append(metric)
            self._weights.append(1.0)

        return self

    def __mul__(self, multiplier: float):
        self._weights = [w * multiplier for w in self._weights]
        return self

    __rmul__ = __mul__


class MultiHorizonMetric(Metric):
    """
    Abstract class for defining metric for a multihorizon forecast
    """

    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__(reduction=reduction, **kwargs)
        self.add_state(
            "losses",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum" if reduction != "none" else "cat",
        )
        self.add_state(
            "lengths",
            default=torch.tensor(0),
            dist_reduce_fx="sum" if reduction != "none" else "mean",
        )

    def loss(
        self, y_pred: Dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss without reduction. Override in derived classes

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: loss/metric as a single number for backpropagation
        """
        raise NotImplementedError()

    def update(self, y_pred, target):
        """
        Update method of metric that handles masking of values.

        Do not override this method but :py:meth:`~loss` instead

        Args:
            y_pred (Dict[str, torch.Tensor]): network output
            target (Union[torch.Tensor, rnn.PackedSequence]): actual values

        Returns:
            torch.Tensor: loss as a single number for backpropagation
        """
        # unpack weight
        if isinstance(target, (list, tuple)) and not isinstance(
            target, rnn.PackedSequence
        ):
            target, weight = target
        else:
            weight = None

        # unpack target
        if isinstance(target, rnn.PackedSequence):
            target, lengths = unpack_sequence(target)
        else:
            lengths = torch.full(
                (target.size(0),),
                fill_value=target.size(1),
                dtype=torch.long,
                device=target.device,
            )

        losses = self.loss(y_pred, target)
        # weight samples
        if weight is not None:
            losses = losses * unsqueeze_like(weight, losses)
        self._update_losses_and_lengths(losses, lengths)

    def _update_losses_and_lengths(self, losses: torch.Tensor, lengths: torch.Tensor):
        losses = self.mask_losses(losses, lengths)
        if self.reduction == "none":
            if self.losses.ndim == 0:
                self.losses = losses
                self.lengths = lengths
            else:
                self.losses = torch.cat([self.losses, losses], dim=0)
                self.lengths = torch.cat([self.lengths, lengths], dim=0)
        else:
            losses = losses.sum()
            if not torch.isfinite(losses):
                losses = torch.tensor(1e9, device=losses.device)
                warnings.warn("Loss is not finite. Resetting it to 1e9")
            self.losses = self.losses + losses
            self.lengths = self.lengths + lengths.sum()

    def compute(self):
        loss = self.reduce_loss(self.losses, lengths=self.lengths)
        return loss

    def mask_losses(
        self, losses: torch.Tensor, lengths: torch.Tensor, reduction: str = None
    ) -> torch.Tensor:
        """
        Mask losses.

        Args:
            losses (torch.Tensor): total loss. first dimenion are samples, second timesteps
            lengths (torch.Tensor): total length
            reduction (str, optional): type of reduction. Defaults to ``self.reduction``.

        Returns:
            torch.Tensor: masked losses
        """
        if reduction is None:
            reduction = self.reduction
        if losses.ndim > 0:
            # mask loss
            mask = torch.arange(losses.size(1), device=losses.device).unsqueeze(
                0
            ) >= lengths.unsqueeze(-1)
            if losses.ndim > 2:
                mask = mask.unsqueeze(-1)
                dim_normalizer = losses.size(-1)
            else:
                dim_normalizer = 1.0
            # reduce to one number
            if reduction == "none":
                losses = losses.masked_fill(mask, float("nan"))
            else:
                losses = losses.masked_fill(mask, 0.0) / dim_normalizer
        return losses

    def reduce_loss(
        self, losses: torch.Tensor, lengths: torch.Tensor, reduction: str = None
    ) -> torch.Tensor:
        """
        Reduce loss.

        Args:
            losses (torch.Tensor): total loss. first dimenion are samples, second timesteps
            lengths (torch.Tensor): total length
            reduction (str, optional): type of reduction. Defaults to ``self.reduction``.

        Returns:
            torch.Tensor: reduced loss
        """
        if reduction is None:
            reduction = self.reduction
        if reduction == "none":
            return losses  # return immediately, no checks
        elif reduction == "mean":
            loss = losses.sum() / lengths.sum()
        elif reduction == "sqrt-mean":
            loss = losses.sum() / lengths.sum()
            loss = loss.sqrt()
        else:
            raise ValueError(f"reduction {reduction} unknown")
        assert not torch.isnan(loss), (
            "Loss should not be nan - i.e. something went wrong "
            "in calculating the loss (e.g. log of a negative number)"
        )
        assert torch.isfinite(
            loss
        ), "Loss should not be infinite - i.e. something went wrong (e.g. input is not in log space)"
        return loss
