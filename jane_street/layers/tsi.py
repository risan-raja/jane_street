import torch
import torch.nn as nn
from .dsattention import DSAttention


class TimeSeriesInteractionNetwork(nn.Module):
    def __init__(
        self,
        num_channels,
        hidden_dim,
        num_blocks,
        edges,
        initial_edge_weights,
        output_dim=4,
        dropout_rate=0.2,
        activation="gelu",
        leaky_relu_slope=0.01,
        num_heads=2,
        mask_flag=False,
    ):
        """
        Initializes the TimeSeriesInteractionNetwork with undirected edges, explicit symmetric parameter sharing, pre-determined edges, shared layers, learnable edge weights and gating, and dropout.

        Args:
            num_channels (int): The number of channels in the input time series.
            hidden_dim (int): The hidden dimension for the processing blocks.
            num_blocks (int): The number of blocks in the time series.
            edges (list of tuples): A list of tuples, where each tuple represents an edge
              between blocks (e.g., [(0, 1), (1, 3), (2, 4)]). The numbers indicate the index of the block
            initial_edge_weights (dict)  A dictionary specifying the initial edge weights. Format: dict[edge tuple]: float
            dropout_rate (float): dropout rate
            activation (str) : type of activation to be used in the network.
            leaky_relu_slope (float): slope of leaky relu if activation is relu
            num_heads (int): number of heads for multihead attention.
            mask_flag (bool) : flag to indicate if masking needs to be applied in attention module
        """
        super(TimeSeriesInteractionNetwork, self).__init__()
        self.num_blocks = num_blocks
        self.channel_dim = num_channels // num_blocks
        self.edges = edges  # Store the pre-determined edges
        self.initial_edge_weights = initial_edge_weights
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.leaky_relu_slope = leaky_relu_slope
        self.num_heads = num_heads
        self.output_dim_per_block = output_dim // num_blocks
        self.mask_flag = mask_flag
        # Create list of processing blocks.
        self.processing_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.channel_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    self.get_activation(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, self.channel_dim),
                )
                for _ in range(self.num_blocks)
            ]
        )

        # Shared Edge attention layers (Only for given features)
        self.attn_dim = (
            self.channel_dim
            if self.channel_dim % num_heads == 0
            else (self.channel_dim // num_heads + 1) * num_heads
        )

        self.shared_attn_linear = nn.Linear(
            self.channel_dim * 2, self.attn_dim * self.num_heads
        )
        self.shared_attn_linear_output = nn.Linear(
            int(self.channel_dim * self.num_heads), self.channel_dim
        )
        self.shared_gate_layer = nn.Sequential(
            nn.Linear(self.channel_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.get_activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.shared_interaction_layer_1 = nn.Sequential(
            nn.Linear(self.channel_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            self.get_activation(),
        )
        self.shared_interaction_layer_2 = nn.Linear(hidden_dim, self.channel_dim * 2)
        self.output_linear = nn.ModuleList(
            [
                nn.Linear(self.channel_dim, self.output_dim_per_block)
                for _ in range(self.num_blocks)
            ]
        )
        # Store initial edge weights as learnable params.
        self.edge_weights = nn.ParameterDict()
        for u, v in edges:
            if (u, v) in self.initial_edge_weights.keys():
                self.edge_weights[str((u, v))] = nn.Parameter(
                    torch.tensor(self.initial_edge_weights[(u, v)])
                )
                self.edge_weights[str((v, u))] = nn.Parameter(
                    torch.tensor(self.initial_edge_weights[(u, v)])
                )

        # Replace MultiheadAttention with DSAttention
        self.multi_head_attention_pooling = DSAttention(
            mask_flag=False,
            attention_dropout=dropout_rate,
            num_heads=num_heads,
            learnable_tau_delta=True,
            enc_in=int(output_dim * 2 / num_heads),
            d_model=self.attn_dim,
        )

        self.global_interaction_linear = nn.Linear(
            self.num_blocks * self.output_dim_per_block,
            self.output_dim_per_block * self.num_blocks,
        )

    def get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "gelu":
            return nn.GELU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU(self.leaky_relu_slope)
        else:
            return nn.ReLU()

    def forward(self, x, attn_mask=None):
        """
        Forward pass of the TimeSeriesInteractionNetwork with edge detection and symmetric parameter sharing.

        Args:
            x (torch.Tensor): Input time series data with shape (B, T, C).

        Returns:
            torch.Tensor: Output time series data with shape (B, T, C).
        """
        B, T, C = x.shape
        blocks = torch.chunk(x, self.num_blocks, dim=2)
        processed_blocks = []
        # Process each block individually
        for block, module in zip(blocks, self.processing_blocks):
            processed_block = module(block)
            processed_blocks.append(processed_block)

        # Interaction (Only for given edges)
        out_blocks = [block.clone() for block in processed_blocks]
        processed_edges = set()  # keep track of edges which have been processed.
        l1_edge_weight = 0  # compute l1 regularization for edge weights

        for u, v in self.edges:
            if (u, v) in processed_edges or (v, u) in processed_edges:
                continue

            concat_blocks_uv = torch.cat(
                [processed_blocks[u], processed_blocks[v]], dim=-1
            )
            concat_blocks_vu = torch.cat(
                [processed_blocks[v], processed_blocks[u]], dim=-1
            )

            gate_uv = self.shared_gate_layer(concat_blocks_uv)
            gate_vu = self.shared_gate_layer(concat_blocks_vu)

            edge_weight_uv = self.edge_weights[str((u, v))]
            edge_weight_vu = self.edge_weights[str((v, u))]

            interaction_out_uv = self.shared_interaction_layer_1(concat_blocks_uv)
            interaction_out_uv = self.shared_interaction_layer_2(
                torch.relu(interaction_out_uv)
            )

            interaction_out_vu = self.shared_interaction_layer_1(concat_blocks_vu)
            interaction_out_vu = self.shared_interaction_layer_2(
                torch.relu(interaction_out_vu)
            )
            # Apply GAT inspired attention, using previous interaction output.
            attn_input_uv = torch.stack(
                [
                    interaction_out_uv[:, :, : self.channel_dim],
                    interaction_out_uv[:, :, self.channel_dim :],
                ],
                dim=2,
            )  # B,T,2,C
            attn_input_vu = torch.stack(
                [
                    interaction_out_vu[:, :, : self.channel_dim],
                    interaction_out_vu[:, :, self.channel_dim :],
                ],
                dim=2,
            )  # B,T,2,C
            B, T, D, C = attn_input_uv.shape
            attn_input_uv = attn_input_uv.view(B, T, D * C)
            attn_input_vu = attn_input_vu.view(B, T, D * C)
            attn_input_uv = self.shared_attn_linear(attn_input_uv)
            attn_input_vu = self.shared_attn_linear(attn_input_vu)
            attn_input_uv = self.shared_attn_linear_output(attn_input_uv)  # B,T,C
            attn_input_vu = self.shared_attn_linear_output(attn_input_vu)  # B,T,C

            # Since the parameters are shared and we are using symmetric aggregation,
            # we use the following shared formulation.
            weighted_interaction = (
                attn_input_uv * edge_weight_uv * gate_uv
                + attn_input_vu * edge_weight_vu * gate_vu
            )
            out_blocks[u] = out_blocks[u] + weighted_interaction
            out_blocks[v] = out_blocks[v] + weighted_interaction
            processed_edges.add((u, v))
            l1_edge_weight += torch.abs(edge_weight_uv).sum()
            l1_edge_weight += torch.abs(edge_weight_vu).sum()
        reduced_out_blocks = []

        for block, output_linear in zip(out_blocks, self.output_linear):
            reduced_out_block = output_linear(block)
            reduced_out_blocks.append(reduced_out_block)
        all_blocks = torch.cat(reduced_out_blocks, dim=2)  # B,T, all output features
        attn_input_all_blocks = all_blocks.reshape(
            B, 1, T, self.num_heads, -1
        )  # B,1,T,H,E
        # print(attn_input_all_blocks[:, 0, :, :, :].shape)
        attn_output_all_blocks, _ = self.multi_head_attention_pooling(
            attn_input_all_blocks[:, 0, :, :, :],
            attn_input_all_blocks[:, 0, :, :, :],
            attn_input_all_blocks[:, 0, :, :, :],
            attn_mask,
        )  # B,T, H, E
        output = self.global_interaction_linear(
            attn_output_all_blocks.reshape(B, T, -1)
        )  # B,T,output features
        # Concatenate and return
        return output, l1_edge_weight
