import torch
from .base import MultiHorizonMetric


class SMAPE(MultiHorizonMetric):
    """
    Symmetric mean absolute percentage. Assumes ``y >= 0``.

    Defined as ``2*(y - y_pred).abs() / (y.abs() + y_pred.abs())``
    """

    def loss(self, y_pred, target_w):
        target = target_w[..., 0]
        y_pred = self.to_prediction(y_pred)
        loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
        return loss


class MAPE(MultiHorizonMetric):
    """
    Mean absolute percentage. Assumes ``y >= 0``.

    Defined as ``(y - y_pred).abs() / y.abs()``
    """

    def loss(self, y_pred, target_w):
        target = target_w[..., 0]
        loss = (self.to_prediction(y_pred) - target).abs() / (target.abs() + 1e-8)
        return loss


class MAE(MultiHorizonMetric):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs()``
    """

    def loss(self, y_pred, target_w):
        target = target_w[..., 0]
        # sample_weight = target_w[...,1]
        loss = (self.to_prediction(y_pred) - target).abs()
        return loss


class ZRMSE(MultiHorizonMetric):
    """
    Sample weighted Zero-mean R-squared score

    Defined as ``1 - (w*(y_pred - target).pow(2)).sum() / target.pow(2).sum()``
    """

    def loss(self, y_pred, target_w):
        target = target_w[..., 0]
        sample_weight = target_w[..., 1]
        y_pred = self.to_prediction(y_pred)
        if sample_weight is None:
            sample_weight = torch.ones_like(target)
        loss = (
            1
            - (sample_weight * (y_pred - target).pow(2)).sum()
            / (sample_weight * target.pow(2)).sum()
        )
        return loss
