from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TimeBatchNorm2d(nn.BatchNorm1d):
    """A batch normalization layer that normalizes over the last two dimensions of a
    sequence in PyTorch, mimicking Keras behavior.

    This class extends nn.BatchNorm1d to apply batch normalization across time and
    feature dimensions.

    Attributes:
        num_time_steps (int): Number of time steps in the input.
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, normalized_shape: tuple[int, int]):
        """Initializes the TimeBatchNorm2d module.

        Args:
            normalized_shape (tuple[int, int]): A tuple (num_time_steps, num_channels)
                representing the shape of the time and feature dimensions to normalize.
        """
        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        """Applies the batch normalization over the last two dimensions of the input tensor.

        Args:
            x (Tensor): A 3D tensor with shape (N, S, C), where N is the batch size,
                S is the number of time steps, and C is the number of channels.

        Returns:
            Tensor: A 3D tensor with batch normalization applied over the last two dims.

        Raises:
            ValueError: If the input tensor is not 3D.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input tensor, but got {x.ndim}D tensor instead."
            )

        # Reshaping input to combine time and feature dimensions for normalization
        x = x.reshape(x.shape[0], -1, 1)

        # Applying batch normalization
        x = super().forward(x)

        # Reshaping back to original dimensions (N, S, C)
        x = x.reshape(x.shape[0], self.num_time_steps, self.num_channels)

        return x


class FeatureMixing(nn.Module):
    """A module for feature mixing with flexibility in normalization and activation.

    This module provides options for batch normalization before or after mixing features,
    uses dropout for regularization, and allows for different activation functions.

    Args:
        sequence_length: The length of the sequences to be transformed.
        input_channels: The number of input channels to the module.
        output_channels: The number of output channels from the module.
        ff_dim: The dimension of the feed-forward network internal to the module.
        activation_fn: The activation function used within the feed-forward network.
        dropout_rate: The dropout probability used for regularization.
        normalize_before: A boolean indicating whether to apply normalization before
            the rest of the operations.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        """Initializes the FeatureMixing module with the provided parameters."""
        super().__init__()

        self.norm_before = (
            norm_type((sequence_length, input_channels))
            if normalize_before
            else nn.Identity()
        )
        self.norm_after = (
            norm_type((sequence_length, output_channels))
            if not normalize_before
            else nn.Identity()
        )

        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_channels)

        self.projection = (
            nn.Linear(input_channels, output_channels)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FeatureMixing module.

        Args:
            x: A 3D tensor with shape (N, C, L) where C is the channel dimension.

        Returns:
            The output tensor after feature mixing.
        """
        x_proj = self.projection(x)

        x = self.norm_before(x)

        x = self.fc1(x)  # Apply the first linear transformation.
        x = self.activation_fn(x)  # Apply the activation function.
        x = self.dropout(x)  # Apply dropout for regularization.
        x = self.fc2(x)  # Apply the second linear transformation.
        x = self.dropout(x)  # Apply dropout again if needed.

        x = x_proj + x  # Add the projection shortcut to the transformed features.

        return self.norm_after(x)


class ConditionalFeatureMixing(nn.Module):
    """Conditional feature mixing module that incorporates static features.

    This module extends the feature mixing process by including static features. It uses
    a linear transformation to integrate static features into the dynamic feature space,
    then applies the feature mixing on the concatenated features.

    Args:
        input_channels: The number of input channels of the dynamic features.
        output_channels: The number of output channels after feature mixing.
        static_channels: The number of channels in the static feature input.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in feature mixing.
        dropout_rate: The dropout probability used in the feature mixing operation.
    """

    def __init__(
        self,
        sequence_length: int,  # Dummy
        input_channels: int,
        output_channels: int,
        static_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.fr_static = nn.Linear(static_channels, output_channels)
        self.fm = FeatureMixing(
            sequence_length,
            input_channels + output_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(
        self, x: torch.Tensor, x_static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies conditional feature mixing using both dynamic and static inputs.

        Args:
            x: A tensor representing dynamic features, typically with shape
               [batch_size, time_steps, input_channels].
            x_static: A tensor representing static features, typically with shape
               [batch_size, static_channels].

        Returns:
            A tuple containing:
            - The output tensor after applying conditional feature mixing.
            - The transformed static features tensor for monitoring or further processing.
        """
        v = self.fr_static(
            x_static
        )  # Transform static features to match output channels.
        v = v.unsqueeze(1).repeat(
            1, x.shape[1], 1
        )  # Repeat static features across time steps.

        return self.fm(
            torch.cat([x, v], dim=-1)
        )  # Apply feature mixing on concatenated features.
        # Return detached static feature for monitoring or further use.


class TimeMixing(nn.Module):
    """Applies a transformation over the time dimension of a sequence.

    This module applies a linear transformation followed by an activation function
    and dropout over the sequence length of the input feature tensor after converting
    feature maps to the time dimension and then back.

    Args:
        input_channels: The number of input channels to the module.
        sequence_length: The length of the sequences to be transformed.
        activation_fn: The activation function to be used after the linear transformation.
        dropout_rate: The dropout probability to be used after the activation function.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        """Initializes the TimeMixing module with the specified parameters."""
        super().__init__()
        self.norm = norm_type((sequence_length, input_channels))
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(sequence_length, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the time mixing operations on the input tensor.

        Args:
            x: A 3D tensor with shape (N, C, L), where C = channel dimension and
                L = sequence length.

        Returns:
            The normalized output tensor after time mixing transformations.
        """
        x_temp = feature_to_time(
            x
        )  # Convert feature maps to time dimension. Assumes definition elsewhere.
        x_temp = self.activation_fn(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        x_res = time_to_feature(x_temp)  # Convert back from time to feature maps.

        return self.norm(
            x + x_res
        )  # Apply normalization and combine with original input.


class MixerLayer(nn.Module):
    """A residual block that combines time and feature mixing for sequence data.

    This module sequentially applies time mixing and feature mixing, which are forms
    of data augmentation and feature transformation that can help in learning temporal
    dependencies and feature interactions respectively.

    Args:
        sequence_length: The length of the input sequences.
        input_channels: The number of input channels to the module.
        output_channels: The number of output channels from the module.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in both time and feature mixing.
        dropout_rate: The dropout probability used in both mixing operations.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        """Initializes the MixLayer with time and feature mixing modules."""
        super().__init__()

        self.time_mixing = TimeMixing(
            sequence_length,
            input_channels,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = FeatureMixing(
            sequence_length,
            input_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
            normalize_before=normalize_before,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MixLayer module.

        Args:
            x: A 3D tensor with shape (N, C, L) to be processed by the mixing layers.

        Returns:
            The output tensor after applying time and feature mixing operations.
        """
        x = self.time_mixing(x)  # Apply time mixing first.
        x = self.feature_mixing(x)  # Then apply feature mixing.

        return x


class ConditionalMixerLayer(nn.Module):
    """Conditional mix layer combining time and feature mixing with static context.

    This module combines time mixing and conditional feature mixing, where the latter
    is influenced by static features. This allows the module to learn representations
    that are influenced by both dynamic and static features.

    Args:
        sequence_length: The length of the input sequences.
        input_channels: The number of input channels of the dynamic features.
        output_channels: The number of output channels after feature mixing.
        static_channels: The number of channels in the static feature input.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in both mixing operations.
        dropout_rate: The dropout probability used in both mixing operations.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        static_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.time_mixing = TimeMixing(
            sequence_length,
            input_channels,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = ConditionalFeatureMixing(
            sequence_length,
            input_channels,
            output_channels=output_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """Forward pass for the conditional mix layer.

        Args:
            x: A tensor representing dynamic features, typically with shape
               [batch_size, time_steps, input_channels].
            x_static: A tensor representing static features, typically with shape
               [batch_size, static_channels].

        Returns:
            The output tensor after applying time and conditional feature mixing.
        """
        x = self.time_mixing(x)  # Apply time mixing first.
        x = self.feature_mixing(x, x_static)  # Then apply conditional feature mixing.

        return x


def time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series tensor to a feature tensor."""
    return x.permute(0, 2, 1)


feature_to_time = time_to_feature


class TSMixerExt(nn.Module):
    """TSMixer model for time series forecasting.

    This model forecasts time series data by integrating historical time series data,
    future known inputs, and static contextual information. It uses a combination of
    conditional feature mixing and mixer layers to process and combine these different
    types of data for effective forecasting.

    Args:
        sequence_length: The length of the input time series sequences.
        prediction_length: The length of the output prediction sequences.
        activation_fn: The name of the activation function to be used.
        num_blocks: The number of mixer blocks in the model.
        dropout_rate: The dropout rate used in the mixer layers.
        input_channels: The number of channels in the historical time series data.
        extra_channels: The number of channels in the extra (future known) inputs.
        hidden_channels: The number of hidden channels used in the mixer layers.
        static_channels: The number of channels in the static feature inputs.
        ff_dim: The inner dimension of the feedforward network in the mixer layers.
        output_channels: The number of output channels for the final output. If None,
                         defaults to the number of input_channels.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: The type of normalization to use. "batch" or "layer".
    """

    def __init__(self, config):
        self.encoder_length = config.encoder_length
        self.decoder_length = config.decoder_length
        self.activation_fn = config.activation_fn
        self.num_blocks = config.num_blocks
        self.dropout_rate = config.dropout_rate
        self.input_channels = config.input_channels
        self.extra_channels = config.extra_channels
        self.hidden_channels = config.hidden_channels
        self.static_channels = config.static_channels
        self.ff_dim = config.ff_dim
        self.output_channels = config.output_channels
        self.normalize_before = config.normalize_before
        self.norm_type = config.norm_type = "layer"
        assert self.static_channels > 0, "static_channels must be greater than 0"
        super().__init__()

        # Transform activation_fn string to callable function
        if hasattr(F, self.activation_fn):
            self.activation_fn = getattr(F, self.activation_fn)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn}")

        # Transform norm_type to callable
        assert self.norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {self.norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if self.norm_type == "batch" else nn.LayerNorm

        self.fc_hist = nn.Linear(self.encoder_length, self.decoder_length)
        self.fc_out = nn.Linear(self.hidden_channels, self.output_channels)

        self.feature_mixing_hist = ConditionalFeatureMixing(
            sequence_length=self.decoder_length,
            input_channels=self.input_channels + self.extra_channels,
            output_channels=self.hidden_channels,
            static_channels=self.static_channels,
            ff_dim=self.ff_dim,
            activation_fn=self.activation_fn,
            dropout_rate=self.dropout_rate,
            normalize_before=self.normalize_before,
            norm_type=norm_type,
        )
        self.feature_mixing_future = ConditionalFeatureMixing(
            sequence_length=self.decoder_length,
            input_channels=self.extra_channels,
            output_channels=self.hidden_channels,
            static_channels=self.static_channels,
            ff_dim=self.ff_dim,
            activation_fn=self.activation_fn,
            dropout_rate=self.dropout_rate,
            normalize_before=self.normalize_before,
            norm_type=norm_type,
        )

        self.conditional_mixer = self._build_mixer(
            self.num_blocks,
            self.hidden_channels,
            self.decoder_length,
            ff_dim=self.ff_dim,
            static_channels=self.static_channels,
            activation_fn=self.activation_fn,
            dropout_rate=self.dropout_rate,
            normalize_before=self.normalize_before,
            norm_type=norm_type,
        )
        self.tanh = nn.Tanh()

    @staticmethod
    def _build_mixer(
        num_blocks: int, hidden_channels: int, decoder_length: int, **kwargs
    ):
        """Build the mixer blocks for the model."""
        channels = [2 * hidden_channels] + [hidden_channels] * (num_blocks - 1)

        return nn.ModuleList(
            [
                ConditionalMixerLayer(
                    input_channels=in_ch,
                    output_channels=out_ch,
                    sequence_length=decoder_length,
                    **kwargs,
                )
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x_extra_future = x["decoder_reals"]
        x_hist = x["encoder_targets"]
        x_extra_hist = x["encoder_reals"]
        if x_extra_hist.shape[1] < self.encoder_length:
            raise ValueError(
                f"Encoder Lengths are less than the encoder length {x_extra_hist.shape}"
            )
        elif x_extra_hist.shape[1] > self.encoder_length:
            # Take only the last encoder_length samples
            x_hist = x_hist[:, -self.encoder_length :, :]
            x_extra_hist = x_extra_hist[:, -self.encoder_length :, :]
        if x["decoder_lengths"].max() < self.decoder_length:
            # Take samples from the encoder kind of like repeating
            extra_length = self.decoder_length - x["decoder_lengths"].max()
            x_extra_future = torch.cat(
                [x["encoder_reals"][:, -extra_length:, :], x["decoder_reals"]], dim=1
            )

        elif x["decoder_lengths"].max() > self.decoder_length:
            # Take only the last decoder_length samples
            x_extra_future = x["decoder_reals"][:, -self.decoder_length :, :]

        # Concatenate historical time series data with additional historical data
        x_hist = torch.cat([x_hist, x_extra_hist], dim=-1)  # B x 242 x 88  | (79 + 9)

        # Transform feature space to time space, apply linear trafo, and convert back
        x_hist_temp = x_hist.permute(0, 2, 1)  # B x 88 x 242
        x_hist_temp = self.fc_hist(x_hist_temp)  # B x 88 x 121
        x_hist = time_to_feature(x_hist_temp)  # B x 121 x 88
        x_static = x["encoder_categoricals"][:, -1, [0, 1, 2, 4]].to(
            torch.float32
        )  # B x 4
        # Apply conditional feature mixing to the historical data
        x_hist = self.feature_mixing_hist(
            x_hist, x_static=x_static
        )  # B x 121 x hidden_channels

        # Apply conditional feature mixing to the future data
        x_future = self.feature_mixing_future(
            x_extra_future, x_static=x_static
        )  # B x 121 x hidden_channels

        # Concatenate processed historical and future data
        X = torch.cat([x_hist, x_future], dim=-1)  # B x 121 x 2*hidden_channels
        # Process the concatenated data through the mixer layers
        for mixing_layer in self.conditional_mixer:
            X = mixing_layer(X, x_static=x_static)

        # Since it is point forecasting we can just take the last value

        # Final linear transformation to produce the forecast
        X = self.fc_out(X)  # B x 121 x 1
        X = (self.tanh(X) * 5.0).unsqueeze(-1)
        X = (
            torch.stack(
                [X[i, self.decoder_length - 1, :] for i in range(X.size(0))],
                dim=0,
            )
            .unsqueeze(-1)
            .clamp(
                min=torch.tensor(-5.0).to(X.device),
                max=torch.tensor(5.0).to(X.device),
            )
        )
        # print(X)
        return X
