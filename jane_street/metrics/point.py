from typing import List, Optional
import torch
from .base import MultiHorizonMetric


class SMAPE(MultiHorizonMetric):
    """
    Symmetric mean absolute percentage. Assumes ``y >= 0``.

    Defined as ``2*(y - y_pred).abs() / (y.abs() + y_pred.abs())``
    """

    def loss(self, y_pred, target_w):
        if target_w.ndim == 4:
            target = target_w.squeeze(-1)[..., 0]
        elif target_w.ndim == 3:
            target = target_w[..., 0]
        else:
            raise ValueError(
                f"Invalid target_w shape: {target_w.shape}\n y_pred shape: {y_pred.shape}"
            )
        # print(y_pred.shape)
        if y_pred.ndim > 1:
            y_pred = y_pred.mean(-1)
        loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
        return loss


class MAPE(MultiHorizonMetric):
    """
    Mean absolute percentage. Assumes ``y >= 0``.

    Defined as ``(y - y_pred).abs() / y.abs()``
    """

    def loss(self, y_pred, target_w):
        if target_w.ndim == 4:
            target = target_w.squeeze(-1)[..., 0]
        elif target_w.ndim == 3:
            target = target_w[..., 0]
        else:
            raise ValueError(
                f"Invalid target_w shape: {target_w.shape}\n y_pred shape: {y_pred.shape}"
            )
        # print(y_pred.shape)
        if y_pred.ndim > 1:
            y_pred = y_pred.mean(-1)
        loss = (y_pred - target).abs() / (target.abs() + 1e-8)
        return loss


class MAE(MultiHorizonMetric):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs()``
    """

    def loss(self, y_pred, target_w):
        if target_w.ndim == 4:
            target = target_w.squeeze(-1)[..., 0]
        elif target_w.ndim == 3:
            target = target_w[..., 0]
        else:
            raise ValueError(
                f"Invalid target_w shape: {target_w.shape}\n y_pred shape: {y_pred.shape}"
            )
        # print(y_pred.shape)
        if y_pred.ndim > 1:
            y_pred = y_pred.mean(-1)
        loss = (y_pred - target).abs()
        return loss


class MSSE(MultiHorizonMetric):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs()``
    """

    def __init__(self, reduction="sum", **kwargs):
        self.reduction = reduction
        super().__init__(reduction="sum", **kwargs)
        self.reduction = reduction

    def to_prediction(self, y_pred):
        b, t, c = y_pred.shape
        if c == 3:
            quantile_vectors = [
                torch.ones((b, t, 1), device=y_pred.device) * q for q in [0.3, 0.5, 0.7]
            ]
            quantileV = torch.cat(quantile_vectors, dim=-1)
            return torch.sum((y_pred * quantileV) / 1.5, dim=-1)
        elif c == 1:
            return y_pred.squeeze(-1)
        else:
            return y_pred.mean(-1)

    def loss(self, y_pred, target_w):
        if target_w.ndim == 4:
            target = target_w.squeeze(-1)[..., 0]
            sample_weight = target_w.squeeze(-1)[..., 1]
        elif target_w.ndim == 3:
            target = target_w[..., 0]
            sample_weight = target_w[..., 1]
        else:
            raise ValueError(
                f"Invalid target_w shape: {target_w.shape}\n y_pred shape: {y_pred.shape}"
            )
        # print(y_pred.shape)
        # if y_pred.ndim > 1:
        #     y_pred = y_pred.mean(-1)
        norm_wts = sample_weight / torch.linalg.norm(sample_weight)
        numerator = (((self.to_prediction(y_pred) - target) ** 2) * norm_wts).sum()
        # denominator = (sample_weight * target).sum()
        return ((numerator) / norm_wts.sum()) ** 0.5


class MSE(MultiHorizonMetric):
    """
    Mean Square error.

    Defined as ``(y_pred - target).abs()``
    """

    def __init__(self, reduction="mean", **kwargs):
        self.reduction = reduction
        super().__init__(reduction="mean", **kwargs)
        self.reduction = reduction

    def to_prediction(self, y_pred):
        b, t, c = y_pred.shape
        if c == 3:
            quantile_vectors = [
                torch.ones((b, t, 1), device=y_pred.device) * q for q in [0.3, 0.5, 0.7]
            ]
            quantileV = torch.cat(quantile_vectors, dim=-1)
            return torch.sum((y_pred * quantileV) / 1.5, dim=-1)
        elif c == 1:
            return y_pred.squeeze(-1)
        else:
            return y_pred.mean(-1)

    def loss(self, y_pred, target_w):
        # print(target_w.shape)
        if target_w.ndim == 4:
            # print("hit")
            target = target_w.squeeze(-1)[..., 0]
            # weights = target_w.squeeze(-1)[..., 1]
        elif target_w.ndim == 3:
            target = target_w[..., 0]
            # weights = target_w[..., 1]
        else:
            raise ValueError(
                f"Invalid target_w shape: {target_w.shape}\n y_pred shape: {y_pred.shape}"
            )
        # print(y_pred.shape)
        # if y_pred.ndim > 1:
        #     y_pred = y_pred.mean(-1)
        loss = (self.to_prediction(y_pred) - target) ** 2
        return loss


class RMSE(MSE):
    """
    Root Mean Square error.

    Defined as ``(y_pred - target).abs()``
    """

    def loss(self, y_pred, target_w):
        loss = super().loss(y_pred, target_w)
        return loss**0.5


class ZRMSS(MultiHorizonMetric):
    """
    Sample weighted Zero-mean R-squared Score

    Defined as ``1 - (w*(y_pred - target).pow(2)).sum() / target.pow(2).sum()``
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Quantile loss

        Args:
            quantiles: quantiles for metric
        """
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        self.quantiles = quantiles
        super().__init__(quantiles=quantiles, **kwargs)
        self.quantiles = quantiles

    @property
    def __name__(self):
        return "zrmss"

    def to_prediction(self, y_pred):
        if y_pred.ndim == 3:
            b, t, c = y_pred.shape
            # print(f"y_pred shape: {y_pred.shape}")
            quantile_vectors = [
                torch.ones((b, t, 1), device=y_pred.device) * q for q in [0.3, 0.5, 0.7]
            ]
            quantileV = torch.cat(quantile_vectors, dim=-1)
            # print(f"quantileV shape: {quantileV.shape}")
            return torch.sum((y_pred * quantileV) / 3, dim=-1)
        else:
            return y_pred

    def loss(self, y_pred, target_w):
        if target_w.ndim == 4:
            target = target_w.squeeze(-1)[..., 0]
            sample_weight = target_w.squeeze(-1)[..., 1]
        elif target_w.ndim == 3:
            target = target_w[..., 0]
            sample_weight = target_w[..., 1]
        else:
            raise ValueError(
                f"Invalid target_w shape: {target_w.shape}\n y_pred shape: {y_pred.shape}"
            )
        # print(y_pred.shape)
        # torch.save({
        #     "y_pred": y_pred,
        #     "target": target,
        #     "sample_weight": sample_weight
        # }, "zrmse.pth")
        if y_pred.ndim > 1:
            y_pred = self.to_prediction(y_pred)
        # y_pred = self.to_prediction(y_pred)
        if sample_weight is None:
            sample_weight = torch.ones_like(target)
        # print(f'y_pred: {y_pred.shape}, target: {target.shape}, sample_weight: {sample_weight.shape}')
        numerator = torch.sum(sample_weight * (target - y_pred) ** 2)
        denominator = torch.sum(target * target**2)
        r_squared = 1 - numerator / denominator
        return -1 * (1 - r_squared)  # negative r_squared
