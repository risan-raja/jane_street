from typing import Union, Tuple
import torch
from torch.nn.utils import rnn


def unpack_sequence(
    sequence: Union[torch.Tensor, rnn.PackedSequence],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack RNN sequence.

    Args:
        sequence (Union[torch.Tensor, rnn.PackedSequence]): RNN packed sequence or tensor of which
            first index are samples and second are timesteps

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of unpacked sequence and length of samples
    """
    if isinstance(sequence, rnn.PackedSequence):
        sequence, lengths = rnn.pad_packed_sequence(sequence, batch_first=True)
        lengths = lengths.to(sequence.device)
    else:
        lengths = torch.ones(
            sequence.size(0), device=sequence.device, dtype=torch.long
        ) * sequence.size(1)
    return sequence, lengths


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[(...,) + (None,) * n_unsqueezes]
