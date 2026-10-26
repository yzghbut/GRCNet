import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.average_ema import average_checkpoints


def select_topk(paths, k):
    if k < 1:
        raise ValueError("k must be positive")
    scored = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        score = checkpoint.get("ema_current_metric_value")
        if score is None:
            score = checkpoint.get("current_metric_value")
        if score is None:
            continue
        score = float(score)
        if not torch.isfinite(torch.tensor(score)):
            continue
        scored.append((score, int(checkpoint.get("epoch", -1)), path))
    if len(scored) < k:
        raise ValueError(
            f"only {len(scored)} checkpoints contain finite online metrics; "
            f"cannot select top-{k}"
        )


    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:k]], scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="experiment directory containing model/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    pattern = os.path.join(args.experiment, "model", "epoch_*.pth")
    paths = sorted(glob.glob(pattern))
    selected, scored = select_topk(paths, args.top_k)
    state_dict = average_checkpoints(selected)
    torch.save(
        {
            "epoch": -1,
            "state_dict": state_dict,
            "averaged_from": selected,
            "selection_metric": "ema_current_metric_value",
            "selection_scores": [
                {"path": path, "mIoU": score, "epoch": epoch}
                for score, epoch, path in scored[: args.top_k]
            ],
        },
        args.output,
    )
    print(f"saved averaged checkpoint: {args.output}")
    for path in selected:
        print(path)


if __name__ == "__main__":
    main()
