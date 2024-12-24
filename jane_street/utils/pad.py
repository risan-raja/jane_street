from typing import Union, Tuple, List
import torch
from torch.nn.utils import rnn
from torch.nn import functional as F


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


def padded_stack(
    tensors: List[torch.Tensor],
    side: str = "right",
    mode: str = "constant",
    value: Union[int, float] = 0,
) -> torch.Tensor:
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.

    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        torch.Tensor: stacked tensor
    """
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            F.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value)
            if full_size - x.size(-1) > 0
            else x
            for x in tensors
        ],
        dim=0,
    )
    return out
