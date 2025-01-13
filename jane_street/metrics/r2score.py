import torch
import torchmetrics
from torchmetrics.utilities.checks import _check_same_shape


class WeightedZeroMeanR2Score(torchmetrics.Metric):
    r"""Computes the sample weighted zero-mean R-squared score (R2) loss.

    This metric computes the weighted R-squared score where the baseline is a
    zero mean. This is useful when you want to measure how well your model
    predicts compared to a simple prediction of zero, taking into account
    sample weights.

    The formula for the weighted zero-mean R-squared score is:

    .. math::
        R^2 = 1 - \frac{\sum_{i=1}^{n} w_i (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} w_i y_i^2}

    where:
        - :math:`y_i` is the ground truth target value for sample `i`.
        - :math:`\hat{y}_i` is the predicted value for sample `i`.
        - :math:`w_i` is the weight for sample `i`.
        - :math:`n` is the total number of samples.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics import WeightedZeroMeanR2Score

        >>> preds = torch.tensor([2.5, 0.5, 2.0, 8.0])
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> weights = torch.tensor([0.5, 1.0, 0.8, 0.2])

        >>> metric = WeightedZeroMeanR2Score()
        >>> metric.update(preds, target, weights)
        >>> metric.compute()
        tensor(0.9489)

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = True

    sum_weighted_squared_error: torch.Tensor
    sum_weighted_target_squared: torch.Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "sum_weighted_squared_error",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_weighted_target_squared",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ) -> None:
        """Updates the state with new predictions, target, and sample weights.

        Args:
            preds: Predicted values (model output).
            target: Ground truth values.
            weights: Sample weights.
        """
        _check_same_shape(preds, target)
        _check_same_shape(preds, weights)

        self.sum_weighted_squared_error += torch.sum(weights * (target - preds) ** 2)
        self.sum_weighted_target_squared += torch.sum(weights * target**2)

    def compute(self) -> torch.Tensor:
        """Computes the weighted zero-mean R-squared score.

        Returns:
            The weighted zero-mean R-squared score.
        """
        if self.sum_weighted_target_squared == 0:
            return torch.tensor(
                0.0, device=self.sum_weighted_squared_error.device
            )  # Avoid division by zero
        return 1.0 - self.sum_weighted_squared_error / self.sum_weighted_target_squared
