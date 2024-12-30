import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    MLP to learn the De-stationary factors for variable sequence lengths.
    """

    def __init__(
        self,
        enc_in,
        max_seq_len,
        hidden_dims,
        hidden_layers,
        output_dim,
        kernel_size=3,
        is_delta=False,
    ):
        super(Projector, self).__init__()

        self.max_seq_len = max_seq_len
        if is_delta:
            self.output_dim = max_seq_len
        else:
            self.output_dim = output_dim

        padding = 1
        self.series_conv = nn.Conv1d(
            in_channels=max_seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        self.is_delta = is_delta
        layers = [
            nn.Linear(2 * enc_in, hidden_dims[0]),
            nn.ReLU(),
        ]  # Modified input size
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [
            nn.Linear(hidden_dims[-1], self.output_dim, bias=False)
        ]  # Modified output size
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E, where S can vary across batches.
        # stats: B x 1 x E
        # y:     B x O, where O is the output dimension (e.g., num_heads for tau/delta).
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        padding_len = self.max_seq_len - seq_len
        x_padded = F.pad(x, (0, 0, 0, padding_len))  # Pad only the sequence dimension
        x_conv = self.series_conv(x_padded)  # B x 1 x E
        x_concat = torch.cat(
            [x_conv, stats], dim=2
        )  # Concatenate along the channel dimension: B x 1 x (E + 1)
        x_concat = x_concat.view(batch_size, -1)  # Flatten: B x (1 * (E + 1))
        y = self.backbone(x_concat)  # B x (O * max_seq_len)
        if self.is_delta:
            y = y[:, :seq_len]  # B x seq_len x O

        return y  # B x seq_len x O


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class DSAttention(nn.Module):
    """De-stationary Attention"""

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        learnable_tau_delta=True,
        num_heads=2,
        max_seq_len=968,
        enc_in=4,
        d_model=16,
    ):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.learnable_tau_delta = learnable_tau_delta
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.enc_in = enc_in
        self.d_model = d_model

        if self.learnable_tau_delta:
            hidden_dims = [self.d_model, self.d_model // 2, self.d_model // 4]
            hidden_layers = len(hidden_dims)
            self.tau_projector = Projector(
                self.enc_in,
                self.max_seq_len,
                hidden_dims,
                hidden_layers,
                1,
                num_heads=num_heads,
            )
            self.delta_projector = Projector(
                self.enc_in,
                self.max_seq_len,
                hidden_dims,
                hidden_layers,
                1,
                num_heads=num_heads,
                is_delta=True,
            )

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / E**0.5
        # stats computation
        if self.learnable_tau_delta:
            # stats = torch.concatenate((torch.mean(values, dim=1).unsqueeze(1), torch.std(values, dim=1).unsqueeze(1)), dim=2)
            # print(f'Stats shape: {stats.shape}')
            mean_enc = values.mean(1, keepdim=True).detach().view(B, 1, H * E)
            # print(f'Mean shape: {mean_enc.shape}')
            # print(f'Mean shape: {mean_enc.shape}')
            std_enc = (
                torch.sqrt(
                    torch.var(values, dim=1, keepdim=True, unbiased=False) + 1e-5
                )
                .detach()
                .view(B, 1, H * E)
            )
            # print(f'Std shape: {std_enc.shape}')
            # print(f'Values shape: {values.permute(0, 2, 1, 3).reshape(B,S,-1).shape}')
            tau = (
                self.tau_projector(
                    values.permute(0, 2, 1, 3).reshape(B, S, -1), std_enc
                )
                .reshape(B, 1, 1, 1)
                .exp()
            )
            delta = self.delta_projector(
                values.permute(0, 2, 1, 3).reshape(B, S, -1), mean_enc
            ).reshape(B, 1, 1, S)

        else:
            tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
            delta = (
                0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
            )  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        # print(f'torch.einsum("blhe,bshe->bhls", queries, keys) {torch.einsum("blhe,bshe->bhls", queries, keys).shape}')
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -torch.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
