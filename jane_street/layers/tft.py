from omegaconf import DictConfig
from typing import Dict
from torch import nn
import torch
from .vsn import VariableSelectionNetwork
from .gated_residual import GatedResidualNetwork
from .gated_linear import GatedLinearUnit
from .norm import AddNorm, GateAddNorm
from .embeddings import MultiEmbedding
from .mha import InterpretableMultiHeadAttention
from .revin import RevIN
from .rnn import LSTM
from ..utils.mixins import TupleOutputMixIn


class TFT(nn.Module, TupleOutputMixIn):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.revin = nn.ModuleDict(
            {
                name: RevIN(1, affine=False)
                if idx != config.target_idx
                else RevIN(1, affine=True)
                for idx, name in enumerate(config.responder_variables)
            }
        )
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=dict(config.embedding_sizes),
            categorical_groups=dict(config.categorical_groups)
            if hasattr(config, "categorical_groups")
            else None,
            x_categoricals=list(config.x_categoricals),
            categorical_groups_name_index=dict(config.categorical_groups_name_index)
            if hasattr(config, "categorical_groups_name_index")
            else None,
        )
        self.embedded_categoricals = self.input_embeddings.names()
        self.prescalers = nn.ModuleDict(
            {k: nn.Linear(1, config.real_hidden_size) for k in config.real_variables}
        )
        static_input_size = {
            name: self.input_embeddings.output_size[name]
            for name in config.static_categoricals  # type: ignore
        }
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_size,
            hidden_size=config.hidden_size,
            input_embedding_flags={name: True for name in config.static_categoricals},
            dropout=config.dropout,
            prescalers=self.prescalers,
        )
        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in config.embedding_sizes  # type: ignore
        }
        encoder_input_sizes.update(
            {
                name: config.real_hidden_size
                for name in config.real_variables + config.responder_variables
            }
        )
        decoder_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in config.embedding_sizes  # type: ignore
        }
        decoder_input_sizes.update(
            {name: config.real_hidden_size for name in config.real_variables}
        )
        if config.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, config.real_hidden_size),
                    output_size=config.real_hidden_size,
                    dropout=config.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, config.real_hidden_size),
                        output_size=config.real_hidden_size,
                        dropout=config.dropout,
                    )
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=config.hidden_size,
            input_embedding_flags={name: True for name in self.embedded_categoricals},
            dropout=config.dropout,
            context_size=config.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not config.share_single_variable_networks
                else self.shared_single_variable_grns  # type: ignore
            ),
        )
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=config.hidden_size,
            input_embedding_flags={name: True for name in self.embedded_categoricals},
            dropout=config.dropout,
            context_size=config.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not config.share_single_variable_networks
                else self.shared_single_variable_grns  # type: ignore
            ),
        )
        # Static Encoder
        self.static_context_variable_selection = GatedResidualNetwork(
            config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            config.dropout,
        )
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            config.dropout,
        )
        # Initial Cell State LSTM
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            config.dropout,
        )

        # Static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size,
        )
        self.lstm_encoder = LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout,
            batch_first=True,
        )
        self.lstm_decoder = LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout,
            batch_first=True,
        )
        self.post_lstm_gate_encoder = GatedLinearUnit(
            config.hidden_size, config.hidden_size, config.dropout
        )
        self.post_lstm_gate_decoder = GatedLinearUnit(
            config.hidden_size, config.hidden_size, config.dropout
        )
        self.post_lstm_add_norm_encoder = AddNorm(
            config.hidden_size, config.hidden_size, trainable_add=True
        )
        self.post_lstm_add_norm_decoder = AddNorm(
            config.hidden_size, config.hidden_size, trainable_add=True
        )
        self.static_enrichment = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size,
        )
        self.multihead_attn = InterpretableMultiHeadAttention(
            n_head=config.n_heads,
            d_model=config.hidden_size,
            dropout=config.dropout,
        )
        self.post_attn_gate_norm = GateAddNorm(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            skip_size=config.hidden_size,
            dropout=config.dropout,
        )
        self.pos_wise_ff = GatedResidualNetwork(
            config.hidden_size,
            config.hidden_size * 4,
            config.hidden_size,
            config.dropout,
        )
        self.pre_output_gate_norm = GateAddNorm(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            skip_size=config.hidden_size,
            dropout=config.dropout,
        )

        self.output_layer = nn.Linear(config.hidden_size, config.output_size)

        #### Metadata for Torch Compile
        self.real_variables = list(self.config.real_variables)
        self.responder_variables = list(self.config.responder_variables)
        self.time_varying_categoricals = list(self.config.time_varying_categoricals)
        self.embedded_categoricals = self.input_embeddings.names()
        self.static_categoricals = list(self.config.static_categoricals)
        self.encoder_variables = (
            self.real_variables + self.responder_variables + self.embedded_categoricals
        )
        self.decoder_variables = self.real_variables + self.embedded_categoricals
        self.lstm_layers = self.config.lstm_layers
        self.causal_attention = self.config.causal_attention
        self.target_name = self.config.target_name

    def create_mask(
        self, size: int | torch.Tensor, lengths: torch.LongTensor, inverse: bool = False
    ) -> torch.BoolTensor:
        """
        Create boolean masks of shape len(lenghts) x size.

        An entry at (i, j) is True if lengths[i] > j.

        Args:
            size (int): size of second dimension
            lengths (torch.LongTensor): tensor of lengths
            inverse (bool, optional): If true, boolean mask is inverted. Defaults to False.

        Returns:
            torch.BoolTensor: mask
        """

        if inverse:  # return where values are
            return torch.arange(size, device=lengths.device).unsqueeze(
                0
            ) < lengths.unsqueeze(-1)  # type: ignore
        else:  # return where no values are
            return torch.arange(size, device=lengths.device).unsqueeze(
                0
            ) >= lengths.unsqueeze(-1)  # type: ignore

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(
        self, encoder_lengths: torch.LongTensor, decoder_lengths: torch.LongTensor
    ):
        """
        Returns causal mask to apply for self-attention layer.
        """
        decoder_length = decoder_lengths.max()
        if self.causal_attention:
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=encoder_lengths.device)  # type: ignore
            # indices for which is predicted
            predict_step = torch.arange(
                0, decoder_length, device=encoder_lengths.device
            )[:, None]  # type: ignore
            # do not attend to steps to self or after prediction
            decoder_mask = (
                (attend_step >= predict_step)
                .unsqueeze(0)
                .expand(encoder_lengths.size(0), -1, -1)
            )
        else:
            # there is value in attending to future forecasts if they are made with knowledge currently
            #   available
            #   one possibility is here to use a second attention layer for future attention (assuming different effects
            #   matter in the future than the past)
            #   or alternatively using the same layer but allowing forward attention - i.e. only
            #   masking out non-available data and self
            decoder_mask = (
                self.create_mask(decoder_length, decoder_lengths)
                .unsqueeze(1)
                .expand(-1, decoder_length, -1)
            )  # type: ignore
        # do not attend to steps where data is padded
        encoder_mask = (
            self.create_mask(encoder_lengths.max(), encoder_lengths)
            .unsqueeze(1)
            .expand(-1, decoder_length, -1)
        )  # type: ignore
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask,
                decoder_mask,
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]):
        encoder_lengths = x["encoder_lengths"].long()
        decoder_lengths = x["decoder_lengths"].long()
        # x_reals = torch.cat([x['encoder_reals'], x['decoder_reals']], dim=1)
        # x_cat = torch.cat([x['encoder_categoricals'], x['decoder_categoricals']], dim=1)
        x_encoder_reals = x["encoder_reals"]
        x_decoder_reals = x["decoder_reals"]
        x_encoder_cat = x["encoder_categoricals"]
        x_decoder_cat = x["decoder_categoricals"]
        enc_timesteps = x_encoder_reals.size(1)
        dec_timesteps = x_decoder_reals.size(1)
        timesteps = enc_timesteps + dec_timesteps
        # max_encoder_length = encoder_lengths.max()

        # Encoder Input Processing
        enc_input_vectors = self.input_embeddings(x_encoder_cat)
        enc_input_vectors.update(
            {
                name: x_encoder_reals[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.real_variables)
            }
        )
        enc_input_vectors.update(
            {
                name: self.revin[name](
                    x["encoder_targets"][..., idx].unsqueeze(-1), mode="norm"
                )
                for idx, name in enumerate(self.responder_variables)
            }
        )
        enc_static_embedding = {
            name: enc_input_vectors[name][:, 0] for name in self.static_categoricals
        }
        enc_static_embedding, enc_static_variable_selection = (
            self.static_variable_selection(enc_static_embedding)
        )
        enc_static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(enc_static_embedding), enc_timesteps
        )
        # Decoder Input Processing
        dec_input_vectors = self.input_embeddings(x_decoder_cat)
        dec_input_vectors.update(
            {
                name: x_decoder_reals[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.real_variables)
            }
        )
        dec_static_embedding = {
            name: dec_input_vectors[name][:, 0] for name in self.static_categoricals
        }
        dec_static_embedding, dec_static_variable_selection = (
            self.static_variable_selection(dec_static_embedding)
        )
        dec_static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(dec_static_embedding), dec_timesteps
        )
        enc_embeddings_varying = {
            name: enc_input_vectors[name] for name in self.encoder_variables
        }
        enc_embeddings_varying, encoder_sparse_weights = (
            self.encoder_variable_selection(
                enc_embeddings_varying, enc_static_context_variable_selection
            )
        )
        dec_embeddings_varying = {
            name: dec_input_vectors[name] for name in self.decoder_variables
        }
        dec_embeddings_varying, decoder_sparse_weights = (
            self.decoder_variable_selection(
                dec_embeddings_varying, dec_static_context_variable_selection
            )
        )
        # static_embedding = torch.cat([enc_static_embedding, dec_static_embedding], dim=1)
        input_hidden = self.static_context_initial_hidden_lstm(
            enc_static_embedding
        ).expand(self.lstm_layers, -1, -1)
        input_cell = self.static_context_initial_cell_lstm(enc_static_embedding).expand(
            self.lstm_layers, -1, -1
        )
        encoder_output, (hidden, cell) = self.lstm_encoder(
            enc_embeddings_varying,
            (input_hidden, input_cell),
            lengths=encoder_lengths,
            enforce_sorted=False,
        )
        decoder_output, _ = self.lstm_decoder(
            dec_embeddings_varying,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(
            lstm_output_encoder, enc_embeddings_varying
        )

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(
            lstm_output_decoder, dec_embeddings_varying
        )

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)
        static_context_enrichment = self.static_context_enrichment(dec_static_embedding)
        dec_attn_input = self.static_enrichment(
            lstm_output_decoder,
            self.expand_static_context(static_context_enrichment, dec_timesteps),
        )
        attn_input = self.static_enrichment(
            lstm_output,
            self.expand_static_context(static_context_enrichment, timesteps),
        )
        attn_output, attn_output_weights = self.multihead_attn(
            q=dec_attn_input,
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(encoder_lengths, decoder_lengths),  # type: ignore
        )
        attn_output = self.post_attn_gate_norm(attn_output, dec_attn_input)
        output = self.pos_wise_ff(attn_output)
        output = self.pre_output_gate_norm(output, lstm_output_decoder)  # B x T x 3
        output = self.revin[self.target_name](output, mode="denorm")
        output = self.output_layer(output)  # This is the last step.
        output = torch.stack(
            [output[i, x["decoder_lengths"][i] - 1, :] for i in range(output.size(0))],
            dim=0,
        ).unsqueeze(-1)
        # print(output.shape)
        # However we need the last decoded step and disregard the padding.
        # print(f'Output Shape is {output.shape}')
        decoder_lengths = torch.ones_like(decoder_lengths)
        return self.to_network_output(
            prediction=output,
            # attention=attn_output_weights,
            # static_variables=torch.cat(
            #     [enc_static_variable_selection, dec_static_variable_selection], dim=1
            # ),
            # encoder_variables=encoder_sparse_weights,
            # decoder_variables=decoder_sparse_weights,
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
        )
