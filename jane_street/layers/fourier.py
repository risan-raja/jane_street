import torch
import torch.nn as nn
from .gfft import INR, RidgeRegressor
from .revin import RevIN
from .star import STAR
from .time_dist import TimeDistributedEmbeddingBag
from .fan import FANLayer


class FourierFastPred(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        # self.n_responders = len(config.responder_variables)
        self.n_responders = 6
        self.n_inr_layers = config.n_inr_layers
        self.inr_hidden_dim = config.inr_hidden_dim
        self.fourier_feature_dim = config.fourier_feature_dim
        self.fourier_scales = config.fourier_scales
        self.resp_embed_dim = config.resp_embed_dim
        self.dropout = config.dropout
        self.total_embed_dim = config.total_embed_dim
        self.n_fan_layers = config.n_fan_layers
        self.num_heads = config.num_heads
        self.output_size = config.output_size
        self.use_norm = config.use_norm
        self.features = config.features
        self.feature_idx = [config.all_reals.index(f) for f in self.features]
        if self.use_norm:
            self.features_revin = RevIN(len(self.features), affine=False)
            self.responders_revin = RevIN(self.n_responders, affine=False)
        self.target_revin = RevIN(1)
        if self.n_responders % self.num_heads != 0:
            raise ValueError(
                f"n_responders ({self.n_responders}) not divisible by num_heads ({self.num_heads})."
            )
        self.enc_inr = INR(
            input_dim=len(self.features),
            n_layers=self.n_inr_layers,
            hidden_dim=self.inr_hidden_dim,
            n_fourier_feats=self.fourier_feature_dim,
            scales=self.fourier_scales,
            dropout=config.dropout,
        )
        self.dec_inr = INR(
            input_dim=len(self.features),
            n_layers=self.n_inr_layers,
            hidden_dim=self.inr_hidden_dim,
            n_fourier_feats=self.fourier_feature_dim,
            scales=self.fourier_scales,
            dropout=config.dropout,
        )
        self.tdem = TimeDistributedEmbeddingBag(
            self.total_embed_dim, self.hidden_dim, batch_first=True
        )
        self.enc_inr_star = STAR(self.hidden_dim * 2, self.hidden_dim)
        self.dec_inr_star = STAR(self.hidden_dim * 2, self.hidden_dim)
        self.resp_embed = FANLayer(self.n_responders, 4)
        self.resp_linear = nn.Linear(4, self.resp_embed_dim)
        # self.fan_layer = FANLayer(1, self.hidden_dim)
        # fan_seq_layers = nn.ModuleList([FANLayer(self.n_responders, self.n_responders) for _ in range(self.n_fan_layers)])
        # fan_seq_layers.append(FANLayer(self.hidden_dim, 1))
        self.fan_layers_1 = FANLayer(self.n_responders, self.hidden_dim)
        self.fan_layers_2 = FANLayer(self.hidden_dim, self.n_responders)
        self.fan_layers = nn.Sequential(
            self.fan_layers_1,
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            self.fan_layers_2,
        )
        # self.fan_layers.append(FANLayer(self.hidden_dim, 1))
        self.regressor = RidgeRegressor(lambda_init=1e-3)
        self.attn = nn.MultiheadAttention(
            3, self.num_heads, batch_first=True, dropout=self.dropout
        )
        self.out_layer = nn.Sequential(
            # nn.Linear(self.n_responders, self.hidden_dim),
            # nn.ReLU(),
            # nn.LayerNorm(self.hidden_dim),
            # nn.Dropout(self.dropout),
            nn.Linear(self.resp_embed_dim, self.output_size),
        )
        # self.out_layer = nn.Linear(self.n_responders, self.output_size)

    def regressor_predict(self, inp, w, b):
        return torch.einsum("... d o, ... t d -> ... t o", [w, inp]) + b

    def forward(self, x: dict[str, torch.Tensor]):
        if self.use_norm:
            x_enc = self.features_revin(
                x["encoder_reals"][..., self.feature_idx], mode="norm"
            )
            x_dec = self.features_revin(
                x["decoder_reals"][..., self.feature_idx], mode="norm"
            )
            x_resp = self.responders_revin(
                x["encoder_targets"][..., [3, 4, 5, 6, 7, 8]], mode="norm"
            )
            # x_resp_2 = self.responders_revin(x["encoder_targets"][..., [3,4,5]], mode="norm")
            # x_resp = x["encoder_targets"]
            _ = self.target_revin(
                x["encoder_targets"][..., 6].unsqueeze(-1), mode="norm"
            )
        else:
            x_enc = x["encoder_reals"]
            x_resp = x["encoder_targets"][..., [3, 4, 5, 6, 7, 8]]
            x_dec = x["decoder_reals"]
            _ = self.target_revin(
                x["encoder_targets"][..., 6].unsqueeze(-1), mode="norm"
            )
        x_enc = self.enc_inr(x_enc[..., self.feature_idx])
        x_enc_cat_emb = self.tdem(x["encoder_categoricals"])
        x_dec = self.dec_inr(x_dec[..., self.feature_idx])
        x_dec_cat_emb = self.tdem(x["decoder_categoricals"])
        # print('Encoding Shape', x_enc.shape, x_dec.shape)
        x_enc = torch.cat([x_enc, x_enc_cat_emb], dim=-1)
        x_dec = torch.cat([x_dec, x_dec_cat_emb], dim=-1)
        x_enc = self.enc_inr_star(x_enc)
        x_dec = self.enc_inr_star(x_dec)
        # print('STAR Shape', x_enc.shape, x_dec.shape)
        resp_fan = self.resp_embed(x_resp)
        resp_fan = self.resp_linear(resp_fan)
        # print("Resp Fans Shape", resp_fans.shape)
        # print("Resp Fans Shape", resp_fan.shape)
        # print("x_resp Shape", x_resp.shape)
        # resp_fan = self.fan_layers(x_resp)
        # print("Resp Fans Shape", resp_fans.shape)
        main_resp = x_resp[..., [0, 1, 2]]
        w, b = self.regressor(x_enc, main_resp - resp_fan)
        # print("Regressor Shape", w.shape, b.shape)
        y_pred = self.regressor_predict(x_dec, w, b)
        # print("Predicted Shape", y_pred.shape)
        attn_out, _ = self.attn(y_pred, main_resp, main_resp)
        # residual = y_pred - attn_out
        y_pred = y_pred + attn_out
        # y_pred = y_pred + attn_out  # Additive Attention
        # print(x_enc.shape, x_dec_inr_star.shape, y_pred.shape)
        y_pred = self.out_layer(y_pred)
        y_pred = torch.stack(
            [y_pred[i, x["decoder_lengths"][i] - 1, :] for i in range(y_pred.size(0))],
            dim=0,
        ).unsqueeze(1)
        # print(y_pred.shape)
        final_pred = self.target_revin(y_pred, mode="denorm").clamp(-5.0, 5.0)
        return final_pred


class FourierIndexPred(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        # self.n_responders = len(config.responder_variables)
        self.lambda_init = config.lambda_init
        self.n_responders = 6
        self.time_indexes_idx = [0, 1]
        self.time_id_idx = [3, 4]
        self.n_inr_layers = config.n_inr_layers
        self.inr_hidden_dim = config.inr_hidden_dim
        self.fourier_feature_dim = config.fourier_feature_dim
        self.fourier_scales = config.fourier_scales
        self.resp_embed_dim = config.resp_embed_dim
        self.dropout = config.dropout
        self.total_embed_dim = config.total_embed_dim
        self.n_fan_layers = config.n_fan_layers
        self.num_heads = config.num_heads
        self.output_size = config.output_size
        self.use_norm = config.use_norm
        self.features = config.features
        self.feature_idx = [config.all_reals.index(f) for f in self.features]
        if self.use_norm:
            # self.features_revin = RevIN(len(self.features), affine=False)
            self.responders_revin = RevIN(self.n_responders, affine=False)
        self.target_revin = RevIN(1)
        if self.n_responders % self.num_heads != 0:
            raise ValueError(
                f"n_responders ({self.n_responders}) not divisible by num_heads ({self.num_heads})."
            )
        self.inr = INR(
            input_dim=self.hidden_dim + len(self.time_indexes_idx),
            n_layers=self.n_inr_layers,
            hidden_dim=self.inr_hidden_dim,
            n_fourier_feats=self.fourier_feature_dim,
            scales=self.fourier_scales,
            dropout=config.dropout,
        )

        self.tdem = TimeDistributedEmbeddingBag(
            self.total_embed_dim, self.hidden_dim, batch_first=True
        )

        self.regressor = RidgeRegressor(lambda_init=1.0)
        self.out_layer = nn.Linear(3, self.output_size)
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def regressor_predict(self, inp, w, b):
        return torch.einsum("... d o, ... t d -> ... t o", [w, inp]) + b

    def forward(self, x: dict[str, torch.Tensor]):
        _ = self.target_revin(x["encoder_targets"][..., 6].unsqueeze(-1), mode="norm")
        if self.use_norm:
            x_resp = self.responders_revin(
                x["encoder_targets"][..., [3, 4, 5, 6, 7, 8]], mode="norm"
            )
        else:
            x_resp = x["encoder_targets"][..., [3, 4, 5, 6, 7, 8]]

        x_enc = x["encoder_reals"][..., self.time_indexes_idx]
        x_dec = x["decoder_reals"][..., self.time_indexes_idx]
        x_enc_cat_emb = self.tdem(x["encoder_categoricals"][..., self.time_id_idx])
        x_dec_cat_emb = self.tdem(x["decoder_categoricals"][..., self.time_id_idx])
        x_enc = torch.cat([x_enc, x_enc_cat_emb], dim=-1)
        x_dec = torch.cat([x_dec, x_dec_cat_emb], dim=-1)
        x_enc = self.inr(x_enc)
        x_dec = self.inr(x_dec)
        main_resp = x_resp[..., [0, 1, 2]]
        w, b = self.regressor(x_enc, main_resp)
        # print("Regressor Shape", w.shape, b.shape)
        y_pred = self.regressor_predict(x_dec, w, b)
        # residual = y_pred - main_resp
        y_pred = self.out_layer(y_pred)
        y_pred = torch.stack(
            [y_pred[i, x["decoder_lengths"][i] - 1, :] for i in range(y_pred.size(0))],
            dim=0,
        ).unsqueeze(1)
        # print(y_pred.shape)
        y_pred = (self.activation(y_pred) * 5.0).clamp(-5.0, 5.0)
        return y_pred
