import torch
from torch.nn.utils import rnn


def custom_collate_fn(
    batches: list[tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]],
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    return (
        dict(
            encoder_lengths=torch.tensor(
                [batch[0]["encoder_length"] for batch in batches], dtype=torch.long
            ),
            decoder_lengths=torch.tensor(
                [batch[0]["decoder_length"] for batch in batches], dtype=torch.long
            ),
            # groups=torch.stack([batch[0]["static_covariates"] for batch in batches]),
            encoder_reals=rnn.pad_sequence(
                [batch[0]["encoder_reals"] for batch in batches], batch_first=True
            ),
            encoder_categoricals=rnn.pad_sequence(
                [batch[0]["encoder_categoricals"] for batch in batches],
                batch_first=True,
            ),
            decoder_reals=rnn.pad_sequence(
                [batch[0]["decoder_reals"] for batch in batches], batch_first=True
            ),
            decoder_categoricals=rnn.pad_sequence(
                [batch[0]["decoder_categoricals"] for batch in batches],
                batch_first=True,
            ),
            encoder_targets=rnn.pad_sequence(
                [batch[0]["encoder_targets"] for batch in batches], batch_first=True
            ),
            # target_scale=torch.stack([batch[0]["target_scale"] for batch in batches]),
        ),
        rnn.pad_sequence([batch[1] for batch in batches], batch_first=True),
    )
