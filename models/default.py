# Reliability and confidence gate the Scene Query logit residual.
import math

import torch
import torch.nn as nn
import torch_scatter
import torch_cluster

from models.losses import build_criteria
from models.utils.structure import Point
from models.utils import offset2batch
from utils.debug_indices import assert_finite, assert_index_range
from .builder import MODELS, build_model
from utils.scene_query import (
    scene_query_semantic_loss,
    scene_query_semantic_target,
)
from litept.geobias import _knn_graph

from models.modules import PointModel, PointSequential
import spconv.pytorch as spconv

import torch.distributed as dist
from tqdm import tqdm
import pointops

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():


            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)

        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)

        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        scene_query_semantic_weight=0.0,
        scene_query_semantic_ignore_index=-1,
        scene_query_logit_fusion_init=0.0,
        scene_query_logit_fusion_max=0.0,
        scene_query_logit_fusion_adaptive=False,
        scene_query_logit_fusion_adaptive_floor=0.25,
        scene_query_logit_fusion_boundary_suppress=False,
        scene_query_logit_fusion_class_bias=False,
        scene_query_logit_fusion_per_class=False,
        scene_query_logit_fusion_class_scale_delta=0.5,
        boundary_refine=False,
        boundary_refine_weight=0.05,
        boundary_refine_scale=0.1,
        boundary_refine_hidden=16,
        boundary_refine_k=8,
        prototype_refine=False,
        prototype_refine_weight=0.05,
        prototype_refine_scale=0.05,
        prototype_refine_temperature=0.2,
        prototype_refine_reliability_gate=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.scene_query_semantic_weight = float(scene_query_semantic_weight)
        if self.scene_query_semantic_weight < 0.0:
            raise ValueError("scene_query_semantic_weight must be non-negative")
        self.scene_query_semantic_ignore_index = int(scene_query_semantic_ignore_index)
        self.scene_query_logit_fusion_max = float(scene_query_logit_fusion_max)
        self.scene_query_logit_fusion_adaptive = bool(
            scene_query_logit_fusion_adaptive
        )
        self.scene_query_logit_fusion_adaptive_floor = float(
            scene_query_logit_fusion_adaptive_floor
        )
        self.scene_query_logit_fusion_boundary_suppress = bool(
            scene_query_logit_fusion_boundary_suppress
        )
        self.scene_query_logit_fusion_class_bias = bool(
            scene_query_logit_fusion_class_bias
        )
        if not 0.0 <= self.scene_query_logit_fusion_adaptive_floor <= 1.0:
            raise ValueError(
                "scene_query_logit_fusion_adaptive_floor must be in [0, 1]"
            )
        self.scene_query_logit_fusion_per_class = bool(
            scene_query_logit_fusion_per_class
        )
        self.scene_query_logit_fusion_class_scale_delta = float(
            scene_query_logit_fusion_class_scale_delta
        )
        self.boundary_refine = bool(boundary_refine)
        self.boundary_refine_weight = float(boundary_refine_weight)
        self.boundary_refine_scale = float(boundary_refine_scale)
        self.boundary_refine_k = int(boundary_refine_k)
        if self.boundary_refine_weight < 0.0 or self.boundary_refine_scale < 0.0:
            raise ValueError("boundary refinement weights must be non-negative")
        if self.boundary_refine_k <= 0:
            raise ValueError("boundary_refine_k must be positive")
        self.prototype_refine = bool(prototype_refine)
        self.prototype_refine_weight = float(prototype_refine_weight)
        self.prototype_refine_scale = float(prototype_refine_scale)
        self.prototype_refine_temperature = float(prototype_refine_temperature)
        self.prototype_refine_reliability_gate = bool(
            prototype_refine_reliability_gate
        )
        if (
            self.prototype_refine_weight < 0.0
            or self.prototype_refine_scale < 0.0
            or self.prototype_refine_temperature <= 0.0
        ):
            raise ValueError("prototype refinement settings must be valid and non-negative")
        if self.prototype_refine:
            self.semantic_prototypes = nn.Parameter(
                torch.empty(num_classes, backbone_out_channels)
            )
            nn.init.normal_(self.semantic_prototypes, mean=0.0, std=0.02)
        else:
            self.register_parameter("semantic_prototypes", None)
        if self.boundary_refine:
            hidden = max(4, int(boundary_refine_hidden))
            self.boundary_head = nn.Sequential(
                nn.Linear(backbone_out_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            self.boundary_residual_head = nn.Sequential(
                nn.Linear(backbone_out_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.boundary_head = None
            self.boundary_residual_head = None
        if (
            not math.isfinite(self.scene_query_logit_fusion_class_scale_delta)
            or not 0.0 <= self.scene_query_logit_fusion_class_scale_delta < 1.0
        ):
            raise ValueError(
                "scene_query_logit_fusion_class_scale_delta must be in [0, 1)"
            )
        fusion_init = float(scene_query_logit_fusion_init)
        if not math.isfinite(fusion_init) or not math.isfinite(
            self.scene_query_logit_fusion_max
        ):
            raise ValueError("scene query fusion settings must be finite")
        if self.scene_query_logit_fusion_max < 0.0:
            raise ValueError("scene_query_logit_fusion_max must be non-negative")
        if abs(fusion_init) > self.scene_query_logit_fusion_max + 1.0e-8:
            raise ValueError(
                "scene_query_logit_fusion_init must be within the fusion maximum"
            )
        if self.scene_query_logit_fusion_max > 0.0:


            normalized_init = fusion_init / self.scene_query_logit_fusion_max
            normalized_init = max(-1.0 + 1.0e-6, min(1.0 - 1.0e-6, normalized_init))
            self.scene_query_logit_fusion = nn.Parameter(
                torch.tensor(math.atanh(normalized_init), dtype=torch.float32)
            )
            if self.scene_query_logit_fusion_per_class or self.scene_query_logit_fusion_class_bias:


                self.scene_query_logit_fusion_class_scale = nn.Parameter(
                    torch.zeros(num_classes, dtype=torch.float32)
                )
            else:
                self.register_parameter("scene_query_logit_fusion_class_scale", None)
        else:
            self.register_parameter("scene_query_logit_fusion", None)
            self.register_parameter("scene_query_logit_fusion_class_scale", None)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):


        key = prefix + "scene_query_logit_fusion"
        if self.scene_query_logit_fusion is not None and key not in state_dict:
            state_dict[key] = torch.zeros_like(self.scene_query_logit_fusion)
        class_scale_key = prefix + "scene_query_logit_fusion_class_scale"
        if (
            self.scene_query_logit_fusion_class_scale is not None
            and class_scale_key not in state_dict
        ):
            state_dict[class_scale_key] = torch.zeros_like(
                self.scene_query_logit_fusion_class_scale
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        input_dict,
        return_point=False,
        return_logits=False,
        logits_only=False,
        return_diagnostics=False,
    ):
        point = Point(input_dict)
        point = self.backbone(point)
        if isinstance(point, Point):
            assert_finite("default_segmentor.backbone.feat", point.feat)


        scene_query_logits_at_query = (
            point.get("scene_query_semantic_logits")
            if isinstance(point, Point)
            else None
        )
        scene_query_loss = None
        scene_query_stats = {}
        if isinstance(point, Point):
            for stat_name in (
                "scene_query_reliability_mean",
                "scene_query_reliability_min",
                "scene_query_reliability_max",
                "scene_query_global_gate_mean",
                "scene_query_local_gate_mean",
            ):
                stat_value = point.get(stat_name)
                if stat_value is not None:
                    scene_query_stats[stat_name] = stat_value.detach()
        if (
            self.training
            and not logits_only
            and self.scene_query_semantic_weight > 0.0
        ):
            if scene_query_logits_at_query is None:
                raise RuntimeError(
                    "scene_query_semantic_weight requires a semantic Scene Query"
                )
            scene_query_target = scene_query_semantic_target(
                point,
                input_dict["segment"],
                scene_query_logits_at_query.shape[1],
                ignore_index=self.scene_query_semantic_ignore_index,
            )
            if scene_query_target.shape[0] != scene_query_logits_at_query.shape[0]:
                raise RuntimeError(
                    "pooled semantic target does not match Scene Query resolution"
                )
            scene_query_loss = scene_query_semantic_loss(
                scene_query_logits_at_query,
                scene_query_target,
                ignore_index=self.scene_query_semantic_ignore_index,
            )


        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                assert_index_range(
                    "default_segmentor.pooling_inverse",
                    inverse,
                    point.feat.shape[0],
                )
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                if "scene_query_semantic_logits" in point.keys():
                    semantic_logits = point.scene_query_semantic_logits
                    assert_index_range(
                        "default_segmentor.scene_query_semantic_logits.inverse",
                        inverse,
                        semantic_logits.shape[0],
                    )
                    parent.scene_query_semantic_logits = semantic_logits[inverse]
                if "geometry_reliability" in point.keys():
                    parent.geometry_reliability = point.geometry_reliability[inverse]

                point = parent
            feat = point.feat
        else:
            feat = point
        assert_finite("default_segmentor.feat", feat)
        seg_logits = self.seg_head(feat)
        prototype_loss = None
        prototype_gate_mean = None
        if self.semantic_prototypes is not None:
            normalized_feat = nn.functional.normalize(feat.float(), dim=-1)
            normalized_prototypes = nn.functional.normalize(
                self.semantic_prototypes.float(), dim=-1
            )
            prototype_logits = (
                normalized_feat @ normalized_prototypes.t()
            ) / self.prototype_refine_temperature
            prototype_gate = 1.0
            if self.prototype_refine_reliability_gate:


                local_confidence = torch.softmax(
                    seg_logits.detach().float(), dim=-1
                ).amax(dim=-1)
                geometry_reliability = (
                    point.get("geometry_reliability")
                    if isinstance(point, Point)
                    else None
                )
                if (
                    geometry_reliability is not None
                    and geometry_reliability.shape[0] == feat.shape[0]
                ):
                    geometry_reliability = geometry_reliability.detach().float().clamp(
                        0.0, 1.0
                    )
                    prototype_gate = local_confidence * geometry_reliability
                else:


                    prototype_gate = local_confidence
                prototype_gate = prototype_gate.clamp(0.0, 1.0).unsqueeze(-1)
                prototype_gate_mean = prototype_gate.mean().detach()
            seg_logits = seg_logits.float() + (
                self.prototype_refine_scale * prototype_gate * prototype_logits
            )
            if self.training and "segment" in input_dict:
                prototype_target = (
                    point.get("segment") if isinstance(point, Point) else None
                )
                if (
                    prototype_target is None
                    or prototype_target.shape[0] != feat.shape[0]
                ):
                    prototype_target = input_dict["segment"]
                if prototype_target.shape[0] == feat.shape[0]:
                    prototype_target = prototype_target.to(feat.device).long()
                    valid = (prototype_target >= 0) & (
                        prototype_target < prototype_logits.shape[1]
                    )
                    if bool(valid.any().item()):
                        prototype_loss = nn.functional.cross_entropy(
                            prototype_logits[valid], prototype_target[valid]
                        )
                    else:
                        prototype_loss = prototype_logits.sum() * 0.0
        boundary_logits = None
        boundary_target = None
        boundary_loss = None
        if self.boundary_head is not None:
            boundary_logits = self.boundary_head(feat).squeeze(-1)
            residual = self.boundary_residual_head(feat)
            boundary_gate = torch.sigmoid(boundary_logits).unsqueeze(-1)
            seg_logits = seg_logits + (
                self.boundary_refine_scale * boundary_gate * residual
            )
            if self.training and "segment" in input_dict:
                labels = point.get("segment") if isinstance(point, Point) else None
                if labels is None or labels.shape[0] != feat.shape[0]:
                    labels = input_dict["segment"]
                if labels.shape[0] == feat.shape[0]:
                    labels = labels.to(feat.device).long()
                    batch = point.batch.long() if isinstance(point, Point) else torch.zeros(
                        feat.shape[0], device=feat.device, dtype=torch.long
                    )
                    with torch.no_grad():
                        edges = _knn_graph(point.coord.float(), batch, self.boundary_refine_k)
                        boundary_target = torch.zeros(
                            feat.shape[0], device=feat.device, dtype=torch.float32
                        )
                        if edges.shape[1] > 0:
                            valid = (labels[edges[0]] >= 0) & (labels[edges[1]] >= 0)
                            edge_boundary = (
                                labels[edges[0]] != labels[edges[1]]
                            ).float() * valid.float()
                            boundary_target.scatter_reduce_(
                                0, edges[0], edge_boundary, reduce="amax", include_self=True
                            )
                    boundary_loss = nn.functional.binary_cross_entropy_with_logits(
                        boundary_logits, boundary_target
                    )
        scene_query_semantic_logits = (
            point.get("scene_query_semantic_logits")
            if isinstance(point, Point)
            else None
        )
        if scene_query_semantic_logits is not None:
            assert_finite(
                "default_segmentor.scene_query_semantic_logits",
                scene_query_semantic_logits,
            )
        fusion_scale = None
        fusion_gate_mean = None
        diagnostic_local_logits = None
        diagnostic_fusion_gate = None
        diagnostic_semantic_prior = None
        if (
            scene_query_semantic_logits is not None
            and self.scene_query_logit_fusion is not None
        ):
            if scene_query_semantic_logits.shape != seg_logits.shape:
                raise ValueError(
                    "Scene Query and segmentation logits must have the same shape"
                )
            fusion_scale = self.scene_query_logit_fusion_max * torch.tanh(
                self.scene_query_logit_fusion
            )
            if self.scene_query_logit_fusion_class_bias:
                class_scale = 1.0 + 0.5 * torch.tanh(
                    self.scene_query_logit_fusion_class_scale
                )
                fusion_scale = fusion_scale * class_scale.to(
                    device=seg_logits.device, dtype=seg_logits.dtype
                )
            fusion_gate = None
            if self.scene_query_logit_fusion_adaptive:


                local_prob = torch.softmax(seg_logits.detach().float(), dim=-1)
                class_count = max(int(seg_logits.shape[1]), 1)
                local_uncertainty = (
                    1.0 - local_prob.max(dim=-1).values
                ) / max(1.0 - 1.0 / class_count, 1.0e-6)
                local_uncertainty = local_uncertainty.clamp(0.0, 1.0)
                geometry_reliability = (
                    point.get("geometry_reliability")
                    if isinstance(point, Point)
                    else None
                )
                if geometry_reliability is not None:
                    geometry_uncertainty = (
                        1.0
                        - geometry_reliability.detach().float().clamp(0.0, 1.0)
                    )
                    uncertainty = torch.maximum(
                        local_uncertainty, geometry_uncertainty
                    )
                else:
                    uncertainty = local_uncertainty
                gate_signal = (
                    1.0 - uncertainty
                    if self.scene_query_logit_fusion_boundary_suppress
                    else uncertainty
                )
                fusion_gate = self.scene_query_logit_fusion_adaptive_floor + (
                    1.0 - self.scene_query_logit_fusion_adaptive_floor
                ) * gate_signal
                fusion_gate = fusion_gate.clamp(0.0, 1.0)
                fusion_gate_mean = fusion_gate.mean().detach()
            semantic_prior = scene_query_semantic_logits.float()
            if self.scene_query_logit_fusion_class_scale is not None:


                prior_mean = semantic_prior.mean(dim=-1, keepdim=True)
                semantic_prior = semantic_prior - prior_mean
                class_scale = 1.0 + self.scene_query_logit_fusion_class_scale_delta * torch.tanh(
                    self.scene_query_logit_fusion_class_scale
                )
                semantic_prior = semantic_prior * class_scale.to(
                    device=semantic_prior.device, dtype=semantic_prior.dtype
                ) + prior_mean


            if return_diagnostics:
                diagnostic_local_logits = seg_logits.float()
                diagnostic_semantic_prior = semantic_prior
                diagnostic_fusion_gate = (
                    torch.ones(
                        seg_logits.shape[0],
                        device=seg_logits.device,
                        dtype=torch.float32,
                    )
                    if fusion_gate is None
                    else fusion_gate.float()
                )


            seg_logits = (
                seg_logits.float()
                + fusion_scale.float()
                * semantic_prior
                * (1.0 if fusion_gate is None else fusion_gate[:, None].float())
            )
        assert_finite("default_segmentor.seg_logits", seg_logits)
        return_dict = dict()
        if return_point:

            return_dict["point"] = point
        if logits_only:
            return_dict["seg_logits"] = seg_logits
            return return_dict

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            if fusion_scale is not None:
                return_dict["scene_query_fusion_scale"] = fusion_scale.detach().mean()
                if fusion_gate_mean is not None:
                    return_dict["scene_query_fusion_gate_mean"] = fusion_gate_mean
            if self.scene_query_semantic_weight > 0.0:
                assert scene_query_loss is not None
                loss = loss + self.scene_query_semantic_weight * scene_query_loss
                return_dict["scene_query_semantic_loss"] = scene_query_loss.detach()
            if boundary_loss is not None:
                loss = loss + self.boundary_refine_weight * boundary_loss
                return_dict["boundary_loss"] = boundary_loss.detach()
            if prototype_loss is not None:
                loss = loss + self.prototype_refine_weight * prototype_loss
                return_dict["prototype_loss"] = prototype_loss.detach()
            if prototype_gate_mean is not None:
                return_dict["prototype_gate_mean"] = prototype_gate_mean
            return_dict["loss"] = loss
            if return_logits:
                return_dict["seg_logits"] = seg_logits
                return_dict.update(scene_query_stats)
                if (
                    isinstance(point, Point)
                    and "geometry_reliability" in point.keys()
                ):
                    return_dict["geometry_reliability"] = point.geometry_reliability

        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
            if return_diagnostics:
                return_dict.update(
                    self._diagnostic_outputs(
                        point,
                        feat,
                        diagnostic_local_logits,
                        seg_logits,
                        diagnostic_semantic_prior,
                        diagnostic_fusion_gate,
                        fusion_scale,
                    )
                )

        else:
            return_dict["seg_logits"] = seg_logits
            if return_diagnostics:
                return_dict.update(
                    self._diagnostic_outputs(
                        point,
                        feat,
                        diagnostic_local_logits,
                        seg_logits,
                        diagnostic_semantic_prior,
                        diagnostic_fusion_gate,
                        fusion_scale,
                    )
                )
        return return_dict

    @staticmethod
    def _diagnostic_outputs(
        point,
        feat,
        local_logits,
        seg_logits,
        semantic_prior,
        fusion_gate,
        fusion_scale,
    ):

        local_logits = (
            seg_logits.float() if local_logits is None else local_logits.float()
        )
        if semantic_prior is None:
            semantic_prior = local_logits.new_zeros(local_logits.shape)
        if fusion_gate is None:
            fusion_gate = local_logits.new_ones(local_logits.shape[0])
        if fusion_scale is None:
            fusion_scale = local_logits.new_zeros(())
        geometry_reliability = (
            point.get("geometry_reliability") if isinstance(point, Point) else None
        )
        if geometry_reliability is None:
            geometry_reliability = local_logits.new_full(
                (local_logits.shape[0],), float("nan")
            )
        else:
            geometry_reliability = geometry_reliability.detach().float().reshape(-1)
        return {
            "diagnostic_local_logits": local_logits,
            "diagnostic_scene_query_prior": semantic_prior.float(),
            "diagnostic_fusion_gate": fusion_gate.float().reshape(-1),
            "diagnostic_fusion_scale": torch.as_tensor(
                fusion_scale, device=local_logits.device
            ).float(),
            "diagnostic_geometry_reliability": geometry_reliability,
            "diagnostic_feature_norm": feat.detach().float().norm(dim=-1),
        }
