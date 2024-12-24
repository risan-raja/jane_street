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


class ZRMSE(MultiHorizonMetric):
    """
    Sample weighted Zero-mean R-squared score

    Defined as ``1 - (w*(y_pred - target).pow(2)).sum() / target.pow(2).sum()``
    """

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
        if y_pred.ndim > 1:
            y_pred = y_pred.mean(-1)
        # y_pred = self.to_prediction(y_pred)
        if sample_weight is None:
            sample_weight = torch.ones_like(target)
        # print(f'y_pred: {y_pred.shape}, target: {target.shape}, sample_weight: {sample_weight.shape}')
        loss = 1 - (
            (sample_weight * (y_pred - target).pow(2)).sum()
            / (sample_weight * target.pow(2)).sum()
        )
        return loss
