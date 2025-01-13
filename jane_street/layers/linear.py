from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def r2_val(y_true, y_pred, sample_weight):
    """
    Calculate weighted R² score
    Args:
        y_true: True values
        y_pred: Predicted values
        sample_weight: Weights for each sample
    Returns:
        Weighted R² score
    """
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (
        np.average((y_true) ** 2, weights=sample_weight) + 1e-38
    )
    return r2


class NN(LightningModule):
    """Neural Network model using PyTorch Lightning"""

    def __init__(
        self,
        input_dim=88,
        hidden_dims=[512, 512, 256],
        dropouts=[0.1, 0.1],
        lr=0.001,
        weight_decay=0.0005,
    ):
        """
        Initialize the neural network
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropouts: List of dropout rates
            lr: Learning rate
            weight_decay: Weight decay for regularization
        """
        super().__init__()
        self.save_hyperparameters()

        # Build network architecture
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(in_dim))  # Batch normalization
            if i > 0:
                layers.append(nn.SiLU())  # SiLU activation (except first layer)
            if i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))  # Dropout for regularization
            layers.append(nn.Linear(in_dim, hidden_dim))  # Linear layer
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())  # Tanh activation for bounded output

        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_step_outputs = []

    def forward(self, x):
        """Forward pass with scaling"""
        return 5 * self.model(x).squeeze(-1)  # Scale output to [-5, 5] range

    def training_step(self, batch):
        """Single training step"""
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction="none") * w  # Weighted MSE loss
        loss = loss.mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch):
        """Single validation step"""
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction="none") * w
        loss = loss.mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.validation_step_outputs.append((y_hat, y, w))
        return loss

    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end"""
        if not self.trainer.sanity_checking:
            y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = (
                torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
            )
            val_r_square = r2_val(y, prob, weights)
            self.log(
                "val_r_square",
                val_r_square,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def on_train_epoch_end(self):
        """Log metrics at end of training epoch"""
        if not self.trainer.sanity_checking:
            epoch = self.trainer.current_epoch
            metrics = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in self.trainer.logged_metrics.items()
            }
            formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
            print(f"Epoch {epoch}: {formatted_metrics}")
