import torch
import torch.nn as nn
from omegaconf import DictConfig
from .gated_residual import GatedResidualNetwork
from .vsn import GroupVSN
from .star import STAR
from .tsi import TimeSeriesInteractionNetwork


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
    def __init__(self, configs: DictConfig):
        super().__init__()
        self.configs = configs
        self.hidden_real_size = configs.hidden_real_size
        self.hidden_cat_size = configs.hidden_cat_size
        self.feature_groups = configs.feature_groups
        self.responder_groups = configs.responder_groups
        self.context_group = configs.context_group
        # self.feature_groups = [g for g in configs.feature_groups] # safety
        self.major_groups = len(self.feature_groups)
        self.features = configs.features
        self.all_reals = configs.all_reals
        self.responders = configs.responders
        self.cat_features = configs.cat_features
        self.dropout = configs.droup_out
        self.group_hidden_size = configs.group_hidden_size
        self.cat_feature_graph = CategoricalFeatureGraph(configs)
        self.context_size = configs.group_hidden_size
        self.num_heads = configs.num_heads
        self.real_value_embeddings = nn.ModuleDict(
            {
                name: GatedResidualNetwork(
                    input_size=1,
                    hidden_size=self.hidden_real_size,
                    output_size=self.hidden_real_size,
                    dropout=self.dropout,
                    residual=False,
                )
                for name in self.all_reals
            }
        )
        # Shared GRNs for real features
        self.feature_grns = nn.ModuleDict(
            {
                name: GatedResidualNetwork(
                    input_size=self.hidden_real_size,
                    hidden_size=self.hidden_real_size,
                    output_size=self.group_hidden_size,
                    dropout=self.dropout,
                    context_size=self.context_size,
                )
                for name in self.all_reals
            }
        )
        # Shared GRNs for categorical features
        self.feature_grns.update(
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
        self.context_group = GroupVSN(
            group_id="context_group",
            features=self.context_group["context_group"],
            input_sizes={
                feature: int(self.cat_feature_graph.embeddings[feature].embedding_dim)
                if feature in self.cat_features
                else self.hidden_real_size
                for feature in self.context_group["context_group"]
            },
            hidden_size=self.hidden_cat_size,
            dropout=self.dropout,
            context_size=self.context_size,
            single_variable_grns=self.feature_grns,
            is_context=True,
        )
        self.feature_group_vsn = nn.ModuleDict(
            {
                group_name: GroupVSN(
                    group_id=group_name,
                    features=group,
                    input_sizes={
                        feature: int(
                            self.cat_feature_graph.embeddings[
                                feature
                            ].embedding_dim  # safety
                        )
                        if feature in self.cat_features
                        else self.hidden_real_size
                        for feature in group
                    },
                    hidden_size=self.hidden_real_size,
                    dropout=self.dropout,
                    context_size=self.context_size,
                    single_variable_grns=self.feature_grns,
                )
                for group_name, group in configs.feature_groups.items()
            }
        )
        self.responder_group_vsn = nn.ModuleDict(
            {
                group_name: GroupVSN(
                    group_id=group_name,
                    features=group,
                    input_sizes={
                        feature: int(
                            self.cat_feature_graph.embeddings[
                                feature
                            ].embedding_dim  # safety
                        )
                        if feature in self.cat_features
                        else self.hidden_real_size
                        for feature in group
                    },
                    hidden_size=self.hidden_real_size,
                    dropout=self.dropout,
                    context_size=self.context_size,
                    single_variable_grns=self.feature_grns,
                )
                for group_name, group in configs.responder_groups.items()
            }
        )
        self.feature_stars = nn.ModuleDict(
            {
                name: STAR(self.group_hidden_size, self.hidden_real_size)
                for name in self.configs.feature_groups
            }
        )
        self.responder_stars = nn.ModuleDict(
            {
                name: STAR(self.group_hidden_size, self.hidden_real_size)
                for name in configs.responder_groups
            }
        )
        self.feature_time_interaction = TimeSeriesInteractionNetwork(
            num_channels=len(self.feature_stars) * self.group_hidden_size,
            hidden_dim=self.hidden_real_size,
            num_blocks=len(self.feature_stars),
            edges=configs.feature_group_edges,
            initial_edge_weights={
                tuple(edge): float(weight)
                for edge, weight in zip(
                    configs.feature_group_edges, configs.feature_group_edge_weights
                )
            },
            output_dim=int(self.hidden_real_size / 2) * len(self.feature_stars),
            dropout_rate=self.dropout,
            activation="gelu",
            leaky_relu_slope=0.01,
            num_heads=self.num_heads,
        )
        self.responder_time_interaction = TimeSeriesInteractionNetwork(
            num_channels=len(self.responder_stars) * self.group_hidden_size,
            hidden_dim=self.hidden_real_size,
            num_blocks=len(configs.responder_groups),
            edges=configs.responder_group_edges,
            initial_edge_weights={
                tuple(edge): float(weight)
                for edge, weight in zip(
                    configs.responder_group_edges, configs.responder_group_edge_weights
                )
            },
            output_dim=int(self.hidden_real_size / 2) * len(self.responder_stars),
            dropout_rate=self.dropout,
            activation="gelu",
            leaky_relu_slope=0.01,
            num_heads=self.num_heads,
        )

        self.lone_groups = configs.lone_groups

    def forward(self, x_real: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        cat_inputs = self.cat_feature_graph(x_cat)
        real_inputs = {
            name: self.real_value_embeddings[name](x_real[..., idx].unsqueeze(-1))
            for idx, name in enumerate(self.all_reals)
        }
        inputs = {**real_inputs, **cat_inputs}
        context, _ = self.context_group(
            {name: inputs[name] for name in self.context_group.features}
        )
        # First pass through all the feature GRNs and store the outputs
        feature_grn_outputs = {
            name: self.feature_grns[name](inputs[name], context) for name in inputs
        }
        feature_groups = {}
        for group_name, group in self.feature_group_vsn.items():
            feature_groups[group_name], _ = group(
                inputs,
                feature_grn_outputs,
                context,
            )
        responder_groups = {}
        for group_name, group in self.responder_group_vsn.items():
            responder_groups[group_name], _ = group(
                inputs,
                feature_grn_outputs,
                context,
            )
        feature_stars = {
            name: self.feature_stars[name](feature_groups[name], context)
            for name in feature_groups
        }
        responder_stars = {
            name: self.responder_stars[name](responder_groups[name], context)
            for name in responder_groups
        }
        # StARS
        feature_stars = torch.cat(list(feature_stars.values()), dim=-1)
        responder_stars = torch.cat(list(responder_stars.values()), dim=-1)
        # TSI
        feature_stars = self.feature_time_interaction(feature_stars)
        responder_stars = self.responder_time_interaction(responder_stars)
        return feature_stars, responder_stars
