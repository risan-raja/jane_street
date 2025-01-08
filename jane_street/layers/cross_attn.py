import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OptimizedCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        output_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_dropout_prob = attention_dropout
        self.output_dropout_prob = output_dropout

        assert (
            self.head_dim * num_heads == d_model
        ), "d_model not divisible by num_heads"

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(output_dropout)
        self.scaling_factor = math.sqrt(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize weights with Xavier initialization for better training stability
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)
        # Initialize biases to zero
        nn.init.zeros_(self.wq.bias)
        nn.init.zeros_(self.wk.bias)
        nn.init.zeros_(self.wv.bias)
        nn.init.zeros_(self.wo.bias)

    def generate_mask(self, decoder_lengths, encoder_lengths):
        """
        Generates a mask for the cross-attention mechanism.

        Args:
            decoder_lengths (torch.Tensor): A tensor of shape (B,) containing the lengths of the decoder sequences.
            encoder_lengths (torch.Tensor): A tensor of shape (B,) containing the lengths of the encoder sequences.

        Returns:
            torch.Tensor: A mask tensor of shape (B, num_heads, T_future, T_lookback) where 1 indicates a valid position and 0 indicates a padded position.
        """
        B = decoder_lengths.size(0)
        max_decoder_len = decoder_lengths.max()
        max_encoder_len = encoder_lengths.max()

        # Create masks for decoder and encoder sequences
        decoder_mask = torch.arange(max_decoder_len).expand(B, max_decoder_len).to(
            decoder_lengths.device
        ) < decoder_lengths.unsqueeze(1)
        encoder_mask = torch.arange(max_encoder_len).expand(B, max_encoder_len).to(
            encoder_lengths.device
        ) < encoder_lengths.unsqueeze(1)

        # Expand masks to match attention scores shape
        decoder_mask = decoder_mask.unsqueeze(1).unsqueeze(
            -1
        )  # (B, 1, max_decoder_len, 1)
        encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(
            -2
        )  # (B, 1, 1, max_encoder_len)

        # Combine masks: valid only if both decoder and encoder positions are valid
        final_mask = (
            decoder_mask & encoder_mask
        )  # (B, 1, max_decoder_len, max_encoder_len)

        # Expand to number of heads
        final_mask = final_mask.expand(
            -1, self.num_heads, -1, -1
        )  # (B, num_heads, max_decoder_len, max_encoder_len)

        return final_mask

    def forward(
        self,
        encoded_representation: torch.Tensor,
        future_queries: torch.Tensor,
        decoder_lengths: torch.Tensor,
        encoder_lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, T_future, D_q = future_queries.shape
        B, T_lookback, D_kv = encoded_representation.shape
        # assert D_q == self.d_model and D_kv == self.d_model, "Input dimension mismatch"

        # Generate mask based on encoder and decoder lengths
        mask = self.generate_mask(decoder_lengths, encoder_lengths)
        # mask = None
        # Project queries, keys, and values
        Q = (
            self.wq(future_queries)
            .reshape(B, T_future, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, num_heads, T_future, head_dim)
        K = (
            self.wk(encoded_representation)
            .reshape(B, T_lookback, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, num_heads, T_lookback, head_dim)
        V = (
            self.wv(encoded_representation)
            .reshape(B, T_lookback, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, num_heads, T_lookback, head_dim)

        # Scaled dot-product attention
        attention_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / self.scaling_factor
        )  # (B, num_heads, T_future, T_lookback)

        # Apply mask if provided
        # if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e-9)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(
            attention_probs
        )  # Apply dropout to attention weights

        attended_values = torch.matmul(
            attention_probs, V
        )  # (B, num_heads, T_future, head_dim)

        # Concatenate heads and project
        attended_values = attended_values.transpose(1, 2).reshape(
            B, T_future, self.d_model
        )
        projected_representation = self.wo(attended_values)
        projected_representation = self.output_dropout(
            projected_representation
        )  # Apply dropout to output

        return projected_representation
