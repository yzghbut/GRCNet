# Distillation uses detached EMA targets after confidence filtering.
import math

import torch
import torch.nn.functional as F


def cosine_ema_distill_weight(
    progress,
    max_weight=0.05,
    final_weight=0.02,
    warmup_ratio=0.1,
    ramp_end_ratio=0.2,
    decay_start_ratio=0.8,
):
    if not 0.0 <= final_weight <= max_weight:
        raise ValueError("EMA distillation weights must satisfy 0 <= final <= max")
    if not 0.0 <= warmup_ratio < ramp_end_ratio <= decay_start_ratio < 1.0:
        raise ValueError("EMA distillation schedule ratios are not ordered")

    progress = min(max(float(progress), 0.0), 1.0)
    if progress <= warmup_ratio:
        return 0.0
    if progress < ramp_end_ratio:
        phase = (progress - warmup_ratio) / (ramp_end_ratio - warmup_ratio)
        return max_weight * 0.5 * (1.0 - math.cos(math.pi * phase))
    if progress <= decay_start_ratio:
        return float(max_weight)

    phase = (progress - decay_start_ratio) / (1.0 - decay_start_ratio)
    return final_weight + (max_weight - final_weight) * 0.5 * (
        1.0 + math.cos(math.pi * phase)
    )


def ema_teacher_due(global_step, interval, weight):
    if interval <= 0:
        raise ValueError("EMA teacher interval must be positive")
    return weight > 0.0 and int(global_step) % int(interval) == 0


def confidence_gated_kl(
    student_logits,
    teacher_logits,
    target,
    confidence_threshold=0.8,
    ignore_index=-1,
    geometry_reliability=None,
    reliability_floor=0.0,
):
    if student_logits.ndim != 2 or teacher_logits.shape != student_logits.shape:
        raise ValueError("student and teacher logits must have matching [N, C] shapes")
    if target.ndim != 1 or target.shape[0] != student_logits.shape[0]:
        raise ValueError("target must have shape [N]")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("EMA teacher confidence threshold must be in [0, 1]")
    if not 0.0 <= float(reliability_floor) <= 1.0:
        raise ValueError("reliability_floor must be in [0, 1]")
    if geometry_reliability is not None:
        if geometry_reliability.ndim != 1 or geometry_reliability.shape[0] != target.shape[0]:
            raise ValueError("geometry_reliability must have shape [N]")

    num_classes = student_logits.shape[1]
    labeled = (target != ignore_index) & (target >= 0) & (target < num_classes)
    teacher_prob = F.softmax(teacher_logits.detach().float(), dim=-1)
    confidence, prediction = teacher_prob.max(dim=-1)
    selected = labeled & prediction.eq(target) & confidence.ge(confidence_threshold)

    labeled_count = labeled.sum()
    coverage = selected.sum().float() / labeled_count.clamp_min(1).float()
    if not bool(selected.any().item()):
        return student_logits.sum() * 0.0, coverage

    point_loss = F.kl_div(
        F.log_softmax(student_logits[selected].float(), dim=-1),
        teacher_prob[selected],
        reduction="none",
    ).sum(dim=-1)
    if geometry_reliability is None:
        loss = point_loss.mean()
    else:
        reliability = geometry_reliability[selected].detach().float().clamp(0.0, 1.0)
        floor = float(reliability_floor)
        reliability = floor + (1.0 - floor) * reliability
        loss = (point_loss * reliability).sum() / reliability.sum().clamp_min(1.0e-12)
    return loss, coverage
