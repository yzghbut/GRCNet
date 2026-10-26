import argparse
from collections import OrderedDict

import torch


def _state_dict(checkpoint):
    state = checkpoint.get("ema_state_dict")
    if state is None:
        state = checkpoint.get("state_dict")
    if state is None:
        raise KeyError("checkpoint must contain ema_state_dict or state_dict")
    return state


def average_checkpoints(paths):
    if len(paths) < 2:
        raise ValueError("at least two checkpoints are required for averaging")
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    states = [_state_dict(checkpoint) for checkpoint in checkpoints]
    keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != keys for state in states[1:]):
        raise ValueError("checkpoint state keys do not match")

    averaged = OrderedDict()
    for key in keys:
        values = [state[key] for state in states]
        first = values[0]
        if any(value.shape != first.shape for value in values[1:]):
            raise ValueError(f"checkpoint tensor shape mismatch for {key}")
        if torch.is_floating_point(first) or torch.is_complex(first):
            if torch.is_complex(first):
                stacked = torch.stack([value.to(dtype=first.dtype) for value in values], dim=0)
            else:
                stacked = torch.stack([value.float() for value in values], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=first.dtype)
        else:


            averaged[key] = values[-1].clone()
    return averaged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()
    state_dict = average_checkpoints(args.checkpoints)
    torch.save(
        {
            "epoch": -1,
            "state_dict": state_dict,
            "averaged_from": list(args.checkpoints),
        },
        args.output,
    )
    print(f"saved averaged checkpoint: {args.output}")


if __name__ == "__main__":
    main()
