import torch
import torch.nn as nn
from omegaconf import DictConfig
from .gated_residual import GatedResidualNetwork
from .time_dist import TimeDistributed
from .vsn import GroupVSN
from .star import STAR

# from .tsi import TimeSeriesInteractionNetwork
from .embeddings import TemporalEmbeddingLayer, TemporalPosEmbeddings
from .cross_attn import OptimizedCrossAttention


class CategoricalFeatureGraph(nn.Module):
    def __init__(self, configs: DictConfig):
        super().__init__()
        self.configs = configs
        self.hidden_cat_size = configs.hidden_cat_size
        self.cat_features = configs.cat_features
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_embeddings=in_size[0], embedding_dim=in_size[1])
                for name, in_size in configs.embedding_sizes.items()
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.cat_inputs = {
            name: self.embeddings[name](x[..., idx])
            for idx, name in enumerate(self.configs.cat_features)
        }
        return self.cat_inputs


class FeatureGraph(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.configs = config
        self.hidden_real_size = config.hidden_real_size
        self.hidden_cat_size = config.hidden_cat_size
        self.feature_groups = config.feature_groups
        self.responder_groups = config.responder_groups
        self.context_group_features = config.context_group
        self.major_groups = len(self.feature_groups)
        self.features = config.features
        self.all_reals = config.all_reals
        self.responders = config.responders
        self.cat_features = config.cat_features
        self.feature_group_idx = {
            group_name: [self.all_reals.index(f) for f in group]
            for group_name, group in self.feature_groups.items()
        }
        self.responder_group_idx = {
            group_name: [self.all_reals.index(f) for f in group]
            for group_name, group in self.responder_groups.items()
        }
        self.dropout = config.droup_out
        self.group_hidden_size = config.group_hidden_size
        self.cat_feature_graph = CategoricalFeatureGraph(config)
        self.feature_group_edge_weights = config.feature_group_edge_weights
        self.temporal_embedding_dim = config.temporal_embedding_dim
        self.context_size = config.group_hidden_size + self.temporal_embedding_dim
        self.num_heads = config.num_heads
        self.time_features = config.time_features
        self.feature_group_edges = config.feature_group_edges
        self.responder_group_edges = config.responder_group_edges
        self.responder_group_edge_weights = config.responder_group_edge_weights
        self.num_projection_layers = config.num_projection_layers
        self.wt_index = self.all_reals.index("weight")
        self.output_size = config.output_size
        self.real_value_embeddings = nn.ModuleDict(
            {
                name: GatedResidualNetwork(
                    input_size=1,
                    hidden_size=self.group_hidden_size,
                    output_size=self.group_hidden_size,
                    dropout=self.dropout,
                    context_size=self.group_hidden_size if name != "weight" else None,
                    residual=False,
                )
                for name in self.all_reals
                if name == "weight"
            }
        )
        self.enc_context_projection = TimeDistributed(
            nn.Linear(self.context_size, self.group_hidden_size, bias=False)
        )
        self.dec_context_projection = TimeDistributed(
            nn.Linear(self.context_size, self.group_hidden_size, bias=False)
        )
        self.enc_feature_group_embeddings = nn.ModuleDict(
            {
                group_name: GatedResidualNetwork(
                    input_size=len(group),
                    hidden_size=self.group_hidden_size,
                    output_size=self.group_hidden_size,
                    dropout=self.dropout,
                    context_size=self.group_hidden_size,
                    residual=False,
                )
                for group_name, group in self.feature_groups.items()
            }
        )
        self.dec_feature_group_embeddings = nn.ModuleDict(
            {
                group_name: GatedResidualNetwork(
                    input_size=len(group),
                    hidden_size=self.group_hidden_size,
                    output_size=self.group_hidden_size,
                    dropout=self.dropout,
                    context_size=self.group_hidden_size,
                    residual=False,
                )
                for group_name, group in self.feature_groups.items()
            }
        )
        self.responder_group_embeddings = nn.ModuleDict(
            {
                group_name: GatedResidualNetwork(
                    input_size=len(group),
                    hidden_size=self.group_hidden_size,
                    output_size=self.group_hidden_size,
                    dropout=self.dropout,
                    context_size=self.group_hidden_size,
                    residual=False,
                )
                for group_name, group in self.responder_groups.items()
            }
        )
        # Shared GRNs for categorical features
        self.feature_grns = nn.ModuleDict(
            {
                name: GatedResidualNetwork(
                    input_size=self.cat_feature_graph.embeddings[name].embedding_dim,
                    hidden_size=self.hidden_cat_size,
                    output_size=self.group_hidden_size,
                    dropout=self.dropout,
                    context_size=self.context_size,
                )
                for name in self.cat_features
            }
        )
        self.feature_grns.update(
            {
                "weight": GatedResidualNetwork(
                    input_size=self.group_hidden_size,
                    hidden_size=self.hidden_real_size,
                    output_size=self.group_hidden_size,
                    dropout=self.dropout,
                    context_size=self.context_size,
                )
            }
        )
        self.enc_cat_context_group = GroupVSN(
            group_id="context_group",
            features=self.context_group_features["context_group"],
            input_sizes={
                feature: int(self.cat_feature_graph.embeddings[feature].embedding_dim)
                if feature in self.cat_features
                else self.group_hidden_size
                for feature in self.context_group_features["context_group"]
            },
            hidden_size=self.hidden_cat_size,
            dropout=self.dropout,
            context_size=None,
            single_variable_grns=self.feature_grns,
            is_context=True,
        )
        self.dec_cat_context_group = GroupVSN(
            group_id="context_group",
            features=self.context_group_features["context_group"],
            input_sizes={
                feature: int(self.cat_feature_graph.embeddings[feature].embedding_dim)
                if feature in self.cat_features
                else self.group_hidden_size
                for feature in self.context_group_features["context_group"]
            },
            hidden_size=self.hidden_cat_size,
            dropout=self.dropout,
            context_size=None,
            single_variable_grns=self.feature_grns,
            is_context=True,
        )
        self.temporal_embedding = TemporalEmbeddingLayer(self.temporal_embedding_dim)
        self.temporal_pos_embedding = TemporalPosEmbeddings(self.temporal_embedding_dim)
        self.enc_feature_stars = nn.ModuleDict(
            {
                name: STAR(self.group_hidden_size, int(self.hidden_real_size / 2))
                for name in self.feature_groups
            }
        )
        self.dec_feature_stars = nn.ModuleDict(
            {
                name: STAR(self.group_hidden_size, int(self.hidden_real_size))
                for name in self.feature_groups
            }
        )
        self.responder_stars = nn.ModuleDict(
            {
                name: STAR(self.group_hidden_size, int(self.hidden_real_size))
                for name in self.responder_groups
            }
        )

        self.dec_projection_layers = nn.ModuleList()
        self.dec_projection_layers.append(
            GatedResidualNetwork(
                input_size=(len(self.dec_feature_stars)) * int(self.group_hidden_size),
                hidden_size=int(self.group_hidden_size),
                output_size=int(self.hidden_real_size / 2)
                * len(self.dec_feature_stars),
                dropout=self.dropout,
                context_size=self.group_hidden_size,
                residual=False,
            )
        )
        self.enc_projection_layers = nn.ModuleList()
        self.enc_projection_layers.append(
            GatedResidualNetwork(
                input_size=(len(self.enc_feature_stars) + len(self.responder_stars))
                * int(self.group_hidden_size),
                hidden_size=int(self.group_hidden_size),
                output_size=int(self.hidden_real_size / 2)
                * len(self.dec_feature_stars),
                dropout=self.dropout,
                context_size=self.group_hidden_size,
                residual=False,
            )
        )
        for _ in range(self.num_projection_layers - 1):
            self.enc_projection_layers.append(
                GatedResidualNetwork(
                    input_size=int(self.hidden_real_size / 2)
                    * len(self.dec_feature_stars),
                    hidden_size=int(self.group_hidden_size),
                    output_size=int(self.hidden_real_size / 2)
                    * len(self.dec_feature_stars),
                    dropout=self.dropout,
                    context_size=self.group_hidden_size,
                    residual=True,
                )
            )
        self.enc_dec_cross = OptimizedCrossAttention(
            int(self.hidden_real_size / 2) * len(self.dec_feature_stars),
            num_heads=self.num_heads,
            attention_dropout=self.dropout,
            output_dropout=self.dropout,
        )
        self.enc_dec_cross_post = nn.Linear(
            int(self.hidden_real_size / 2) * len(self.dec_feature_stars),
            int(self.hidden_real_size),
        )
        self.output_projection = nn.ModuleList()
        self.output_projection.append(
            GatedResidualNetwork(
                input_size=int(self.hidden_real_size / 2) * len(self.dec_feature_stars),
                hidden_size=int(self.hidden_real_size),
                output_size=int(self.hidden_real_size),
                dropout=self.dropout,
                context_size=int(self.hidden_real_size),
                residual=False,
            )
        )
        for _ in range(self.num_projection_layers - 1):
            self.output_projection.append(
                GatedResidualNetwork(
                    input_size=int(self.hidden_real_size),
                    hidden_size=int(self.hidden_real_size),
                    output_size=int(self.hidden_real_size),
                    dropout=self.dropout,
                    context_size=int(self.hidden_real_size),
                    residual=True,
                )
            )
        self.output_layer = nn.Linear(int(self.hidden_real_size), self.output_size)

    def generate_padding_mask(self, lengths):
        """
        Generates a padding mask for the TSI attention mechanism for the decoder.

        Args:
            lengths (torch.Tensor): A tensor of shape (B,) containing the lengths of the sequences.
            max_len (int, optional): The maximum length to use for the mask. If None, uses the maximum length in `lengths`.

        Returns:
            torch.Tensor: A mask tensor of shape (B, 1, 1, max_len) where 1 indicates a valid position and 0 indicates a padded position.
        """
        B = lengths.size(0)
        max_len = lengths.max()

        # Create a mask for padded positions
        mask = torch.arange(max_len, device=lengths.device).expand(
            B, max_len
        ) < lengths.unsqueeze(1)  # (B, max_len)

        # Expand to (B, 1, 1, max_len) for compatibility with attention scores
        mask = mask.unsqueeze(1)  # (B, 1, 1, max_len)
        mask = mask.unsqueeze(2).expand(
            B, self.num_heads, max_len, max_len
        )  # (B, num_heads, max_len, max_len)

        return mask

    def forward(
        self,
        x: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        enc_means_reals = x["encoder_reals"].mean(1, keepdim=True).detach()
        x["encoder_reals"] = x["encoder_reals"] - enc_means_reals
        enc_stdev_reals = torch.sqrt(
            torch.var(x["encoder_reals"], dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x["encoder_reals"] /= enc_stdev_reals

        dec_means_reals = x["decoder_reals"].mean(1, keepdim=True).detach()
        x["decoder_reals"] = x["decoder_reals"] - dec_means_reals
        dec_stdev_reals = torch.sqrt(
            torch.var(x["decoder_reals"], dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x["decoder_reals"] /= dec_stdev_reals

        enc_targets_means = x["encoder_targets"].mean(1, keepdim=True).detach()
        x["encoder_targets"] = x["encoder_targets"] - enc_targets_means
        enc_targets_stdev = torch.sqrt(
            torch.var(x["encoder_targets"], dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x["encoder_targets"] /= enc_targets_stdev

        x["encoder_reals"] = torch.cat(
            [x["encoder_reals"], x["encoder_targets"]], dim=-1
        )
        enc_inputs = {
            name: cat_emb
            for name, cat_emb in self.cat_feature_graph(
                x["encoder_categoricals"]
            ).items()
        }
        enc_inputs.update(
            {
                "weight": self.real_value_embeddings["weight"](
                    x["encoder_reals"][..., self.wt_index].unsqueeze(-1)
                )
            }
        )

        dec_inputs = {
            name: cat_emb
            for name, cat_emb in self.cat_feature_graph(
                x["decoder_categoricals"]
            ).items()
        }
        dec_inputs.update(
            {
                "weight": self.real_value_embeddings["weight"](
                    x["decoder_reals"][..., self.wt_index].unsqueeze(-1)
                )
            }
        )

        # Temporal embeddings
        enc_temporal_embedding = {
            name: x["encoder_reals"][..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.all_reals)
            if name in self.time_features
        }
        dec_temporal_embedding = {
            name: x["decoder_reals"][..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.all_reals)
            if name in self.time_features
        }
        enc_temporal_embedding = self.temporal_embedding(
            *self.temporal_pos_embedding(
                enc_temporal_embedding["date_id"],
                enc_temporal_embedding["time_idx"],
            )
        )
        dec_temporal_embedding = self.temporal_embedding(
            *self.temporal_pos_embedding(
                dec_temporal_embedding["date_id"],
                dec_temporal_embedding["time_idx"],
            )
        )

        enc_context, _ = self.enc_cat_context_group(
            {name: enc_inputs[name] for name in self.enc_cat_context_group.features}
        )
        enc_context = torch.cat([enc_context, enc_temporal_embedding], dim=-1)

        enc_context = self.enc_context_projection(enc_context)

        dec_context, _ = self.dec_cat_context_group(
            {name: dec_inputs[name] for name in self.dec_cat_context_group.features}
        )
        # print(enc_context.shape)
        dec_context = torch.cat([dec_context, dec_temporal_embedding], dim=-1)

        dec_context = self.dec_context_projection(dec_context)

        enc_feature_groups = {
            name: self.enc_feature_group_embeddings[name](
                x["encoder_reals"][..., group_idx], enc_context
            )
            for name, group_idx in self.feature_group_idx.items()
        }

        dec_feature_groups = {
            name: self.dec_feature_group_embeddings[name](
                x["decoder_reals"][..., group_idx], dec_context
            )
            for name, group_idx in self.feature_group_idx.items()
        }

        responder_groups = {
            name: self.responder_group_embeddings[name](
                x["encoder_reals"][..., group_idx], enc_context
            )
            for name, group_idx in self.responder_group_idx.items()
        }

        # STARS

        enc_feature_groups = {
            name: self.enc_feature_stars[name](enc_feature_groups[name], enc_context)
            for name in enc_feature_groups
        }
        enc_feature_groups = torch.cat(list(enc_feature_groups.values()), dim=-1)

        dec_feature_groups = {
            name: self.dec_feature_stars[name](dec_feature_groups[name], dec_context)
            for name in dec_feature_groups
        }
        dec_feature_groups = torch.cat(list(dec_feature_groups.values()), dim=-1)

        responder_groups = {
            name: self.responder_stars[name](responder_groups[name], enc_context)
            for name in responder_groups
        }
        responder_groups = torch.cat(list(responder_groups.values()), dim=-1)
        enc_feature_groups = torch.cat([enc_feature_groups, responder_groups], dim=-1)
        for layer in self.enc_projection_layers:
            enc_feature_groups = layer(enc_feature_groups, enc_context)

        for layer in self.dec_projection_layers:
            dec_feature_groups = layer(dec_feature_groups, dec_context)

        # Temporal Cross Attention
        enc_feature_groups = self.enc_dec_cross(
            enc_feature_groups,
            dec_feature_groups,
            x["decoder_lengths"],
            x["encoder_lengths"],
        )
        enc_feature_groups = self.enc_dec_cross_post(enc_feature_groups)

        # Output projection
        for layer in self.output_projection:
            dec_feature_groups = layer(dec_feature_groups, enc_feature_groups)

        dec_feature_groups = torch.stack(
            [
                dec_feature_groups[i, x["decoder_lengths"][i] - 1, :]
                for i in range(dec_feature_groups.size(0))
            ],
            dim=0,
        ).unsqueeze(1)

        dec_feature_groups = self.output_layer(dec_feature_groups).clamp(
            min=torch.tensor(-5.0).to(dec_feature_groups.device),
            max=torch.tensor(5.0).to(dec_feature_groups.device),
        )
        return dec_feature_groups
