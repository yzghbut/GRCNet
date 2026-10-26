import torch
import torch.nn.functional as F


@torch.no_grad()
def scene_query_semantic_target(point, target, num_classes, ignore_index=-1):

    if not torch.is_tensor(target):
        raise TypeError("target must be a tensor")
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")

    target = target.long().view(-1)
    classes = int(num_classes)
    valid = (target != int(ignore_index)) & (target >= 0) & (target < classes)
    pooled = target.new_zeros((target.shape[0], classes), dtype=torch.float32)
    if bool(valid.any().item()):
        pooled[valid, target[valid]] = 1.0


    chain = []
    current = point
    while hasattr(current, "keys") and "pooling_parent" in current.keys():
        if "pooling_inverse" not in current.keys():
            raise KeyError("pooling_parent requires pooling_inverse")
        inverse = current.pooling_inverse.long().view(-1)
        child_size = int(current.feat.shape[0])
        chain.append((inverse, child_size))
        current = current.pooling_parent

    for inverse, child_size in reversed(chain):
        if inverse.shape[0] != pooled.shape[0]:
            raise ValueError(
                "pooling_inverse length must match the current target size"
            )
        if inverse.numel() > 0 and (
            int(inverse.min().item()) < 0 or int(inverse.max().item()) >= child_size
        ):
            raise IndexError("pooling_inverse contains an out-of-range child index")
        next_target = pooled.new_zeros((child_size, classes))
        next_target.index_add_(0, inverse, pooled)
        counts = torch.bincount(inverse, minlength=child_size).to(next_target.dtype)
        pooled = next_target / counts.clamp_min(1).unsqueeze(1)

    return pooled


def scene_query_semantic_loss(logits, target, ignore_index=-1):

    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("scene query semantic logits must have shape [N, C]")
    if not torch.is_tensor(target):
        raise TypeError("target must be a tensor")
    if target.ndim == 2:
        if target.shape != logits.shape:
            raise ValueError(
                "soft semantic target must have the same shape as logits"
            )
        target = target.to(device=logits.device, dtype=torch.float32)
        valid = target.sum(dim=1) > 0
        if not bool(valid.any().item()):
            return logits.sum() * 0.0
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        log_prob = F.log_softmax(logits[valid].float(), dim=-1)
        return -(target[valid] * log_prob).sum(dim=-1).mean()

    target = target.long().view(-1)
    if logits.shape[0] != target.shape[0]:
        raise ValueError("scene query semantic logits and target must have matching N")
    valid = (target != int(ignore_index)) & (target >= 0) & (
        target < logits.shape[1]
    )
    if not bool(valid.any().item()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid].float(), target[valid])
