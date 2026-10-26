import os

import torch


_TRUE_VALUES = {"1", "true", "yes", "on", "debug"}


def debug_indices_enabled():
    value = os.environ.get("LITEPT_DEBUG_INDICES", "")
    return value.strip().lower() in _TRUE_VALUES


def assert_index_range(name, index, size):
    if not debug_indices_enabled():
        return
    if not torch.is_tensor(index):
        raise RuntimeError(
            f"[LITEPT_DEBUG_INDICES] {name} is not a tensor: {type(index)!r}"
        )
    if (
        index.dtype == torch.bool
        or torch.is_floating_point(index)
        or torch.is_complex(index)
    ):
        raise RuntimeError(
            f"[LITEPT_DEBUG_INDICES] {name} must be an integer index tensor; "
            f"got dtype={index.dtype}, shape={tuple(index.shape)}, device={index.device}"
        )

    size = int(size)
    if size < 0:
        raise RuntimeError(f"[LITEPT_DEBUG_INDICES] {name} has negative size={size}")
    if index.numel() == 0:
        return

    detached = index.detach()
    minimum = int(detached.amin().item())
    maximum = int(detached.amax().item())
    if minimum < 0 or maximum >= size:
        valid = "empty only" if size == 0 else f"0..{size - 1}"
        raise RuntimeError(
            f"[LITEPT_DEBUG_INDICES] {name} out of bounds: valid={valid}, "
            f"shape={tuple(index.shape)}, dtype={index.dtype}, device={index.device}, "
            f"min={minimum}, max={maximum}, size={size}"
        )


def assert_finite(name, tensor):
    if not debug_indices_enabled():
        return
    if not torch.is_tensor(tensor):
        raise RuntimeError(
            f"[LITEPT_DEBUG_INDICES] {name} is not a tensor: {type(tensor)!r}"
        )
    if tensor.numel() == 0:
        return

    detached = tensor.detach()
    finite = torch.isfinite(detached)
    if bool(finite.all().item()):
        return

    nan_count = int(torch.isnan(detached).sum().item())
    inf_count = int(torch.isinf(detached).sum().item())
    finite_values = detached[finite]
    if finite_values.numel() == 0:
        minimum = "none"
        maximum = "none"
    else:
        minimum = float(finite_values.amin().item())
        maximum = float(finite_values.amax().item())
    raise RuntimeError(
        f"[LITEPT_DEBUG_INDICES] {name} has non-finite values: "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
        f"nan={nan_count}, inf={inf_count}, finite_min={minimum}, finite_max={maximum}"
    )
