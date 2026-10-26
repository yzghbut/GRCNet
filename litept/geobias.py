# Geometry reliability bounds the low-rank attention bias.
import math
from contextlib import nullcontext
from collections.abc import Sequence

import torch
import torch.nn as nn

from utils.debug_indices import assert_finite, assert_index_range


def _has_point_attr(point, key):
    if hasattr(point, "keys"):
        return key in point.keys() or key in getattr(point, "__dict__", {})
    return key in getattr(point, "__dict__", {})


def _get_point_attr(point, key):
    if hasattr(point, "keys") and key in point.keys():
        return point[key]
    return getattr(point, key)


def _set_point_attr(point, key, value):
    if hasattr(point, "__setitem__"):
        point[key] = value
    else:
        setattr(point, key, value)


def _autocast_disabled(device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=False)
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _point_coordinates(point):
    return _coord_source(point).float()


def _coord_source(point):
    if _has_point_attr(point, "coord"):
        return _get_point_attr(point, "coord")
    return _get_point_attr(point, "grid_coord")


def _point_batch(point, coord):
    if _has_point_attr(point, "batch"):
        return _get_point_attr(point, "batch").long()
    return torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)


def _coord_signature(tensor):
    return (
        tuple(tensor.shape),
        str(tensor.device),
        tensor.data_ptr(),
        tensor._version,
    )


def _cached_geometry(point, cache_key, signature):
    if _has_point_attr(point, cache_key):
        cached_signature, cached_value = _get_point_attr(point, cache_key)
        if cached_signature == signature:
            return cached_value
    return None


def _store_geometry(point, cache_key, signature, value):
    _set_point_attr(point, cache_key, (signature, value))


def _valid_point_index(index, num_points):
    if index.dim() != 1 or index.numel() != num_points:
        return False
    if num_points == 0:
        return True
    in_bounds = (index >= 0) & (index < num_points)
    return bool(in_bounds.all().item())


def _candidate_order_signature(name, tensor):
    return (name, _coord_signature(tensor))


def _fallback_knn_graph(coord, batch, k):
    rows = []
    cols = []
    for batch_id in torch.unique(batch):
        index = torch.nonzero(batch == batch_id, as_tuple=False).flatten()
        if index.numel() <= 1:
            continue
        coord_b = coord[index]
        k_b = min(int(k), int(index.numel()) - 1)
        dist = torch.cdist(coord_b, coord_b)
        dist.fill_diagonal_(float("inf"))
        nn_index = dist.topk(k_b, dim=1, largest=False).indices
        rows.append(index[:, None].expand(-1, k_b).reshape(-1))
        cols.append(index[nn_index.reshape(-1)])
    if not rows:
        empty = torch.empty(0, device=coord.device, dtype=torch.long)
        return torch.stack([empty, empty], dim=0)
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


def _knn_graph(coord, batch, k):
    if coord.shape[0] <= 1:
        empty = torch.empty(0, device=coord.device, dtype=torch.long)
        return torch.stack([empty, empty], dim=0)
    bincount = torch.bincount(batch)
    min_count = int(bincount[bincount > 0].min().item())
    if min_count <= 1:
        return _fallback_knn_graph(coord, batch, k)
    k_eff = min(int(k), min_count - 1)
    try:
        from torch_cluster import knn_graph

        return knn_graph(
            coord,
            k=k_eff,
            batch=batch,
            loop=False,
            flow="target_to_source",
        )
    except Exception:
        return _fallback_knn_graph(coord, batch, k_eff)


def _scatter_mean(src, index, dim_size):
    out = src.new_zeros((dim_size,) + src.shape[1:])
    if src.numel() == 0:
        return out
    assert_index_range("geobias._scatter_mean.index", index, dim_size)
    out.index_add_(0, index, src)
    count = src.new_zeros(dim_size)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / count.clamp_min(1.0).view(-1, *([1] * (src.dim() - 1)))


def _normalize_k_values(k):
    if isinstance(k, Sequence) and not isinstance(k, (str, bytes)):
        values = tuple(int(value) for value in k)
    else:
        values = (int(k),)
    if not values:
        raise ValueError("GeoBias k must contain at least one neighborhood size")
    if any(value <= 0 for value in values):
        raise ValueError("GeoBias k values must be positive integers")
    return values


def _stage_value(value, stage_index, default):
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 0:
            return default
        if stage_index is None:
            return default
        index = max(0, min(int(stage_index), len(value) - 1))
        return value[index]
    return value


@torch.no_grad()
def manifold_geometry_components(point, k=8):
    raw_coord = _coord_source(point)
    signature = _coord_signature(raw_coord)
    cache_key = f"geobias_components_k{k}"
    cached = _cached_geometry(point, cache_key, signature)
    if cached is not None:
        return cached

    coord = raw_coord.float()
    batch = _point_batch(point, coord)
    num_points = coord.shape[0]
    if num_points == 0:
        empty = coord.new_zeros((0,))
        components = (empty, empty, empty, empty)
        _store_geometry(point, cache_key, signature, components)
        return components

    edge_index = _knn_graph(coord, batch, k)
    center = edge_index[0]
    neighbor = edge_index[1]
    assert_index_range("geobias.manifold.center", center, num_points)
    assert_index_range("geobias.manifold.neighbor", neighbor, num_points)
    vector = coord[neighbor] - coord[center]

    metric = _scatter_mean(torch.norm(vector, dim=-1), center, num_points)
    laplacian = _scatter_mean(vector, center, num_points)
    curvature = torch.norm(laplacian, dim=-1)
    assert_finite("geobias.manifold.metric", metric)
    assert_finite("geobias.manifold.laplacian", laplacian)
    assert_finite("geobias.manifold.curvature", curvature)

    assert_index_range("geobias.manifold.jerk_neighbor", neighbor, laplacian.shape[0])
    assert_index_range("geobias.manifold.jerk_center", center, laplacian.shape[0])
    jerk_vector = laplacian[neighbor] - laplacian[center]
    torsion = _scatter_mean(torch.norm(jerk_vector, dim=-1), center, num_points)
    assert_finite("geobias.manifold.torsion", torsion)

    tangent_alignment = torch.norm(laplacian, dim=-1) / metric.clamp_min(1.0e-6)
    confidence = tangent_alignment.clamp(0.0, 1.0).to(coord.dtype)
    assert_finite("geobias.manifold.confidence", confidence)

    components = (metric, curvature, torsion, confidence)
    _store_geometry(point, cache_key, signature, components)
    return components


def _percloud_offsets(value, batch):
    num_clouds = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_clouds == 0:
        return value
    assert_index_range("geobias._percloud_offsets.batch", batch, num_clouds)
    minimum = torch.full(
        (num_clouds,),
        float("inf"),
        device=value.device,
        dtype=value.dtype,
    )
    minimum.scatter_reduce_(0, batch, value, reduce="amin", include_self=True)
    return value - minimum[batch]


def _lexicographic_order(coord, batch):
    key = (
        coord[:, 2]
        + coord[:, 1] * 1.0e4
        + coord[:, 0] * 1.0e8
        + batch.float() * 1.0e12
    )
    return torch.argsort(key)


def _canonical_order(point, coord, batch):
    num_points = coord.shape[0]
    if _has_point_attr(point, "serialized_order_canonical"):
        source = _get_point_attr(point, "serialized_order_canonical")
        candidate = source.to(device=coord.device, dtype=torch.long)
        if _valid_point_index(candidate, num_points):
            return candidate, _candidate_order_signature("canonical", source)

    if _has_point_attr(point, "serialized_order"):
        source = _get_point_attr(point, "serialized_order")
        if source.dim() > 1:
            source = source[0]
        candidate = source.to(device=coord.device, dtype=torch.long)
        if _valid_point_index(candidate, num_points):
            return candidate, _candidate_order_signature("serialized", source)

    return _lexicographic_order(coord, batch), ("lexicographic",)


@torch.no_grad()
def canonical_arclength(point, k_ref=8, clip_ratio=3.0, eps=1.0e-6):
    raw_coord = _coord_source(point)
    coord = raw_coord.float()
    batch = _point_batch(point, coord)
    num_points = coord.shape[0]
    permutation, order_signature = _canonical_order(point, coord, batch)
    signature = (
        _coord_signature(raw_coord),
        order_signature,
        float(clip_ratio),
    )
    cache_key = f"geobias_canonical_s_k{k_ref}"
    cached = _cached_geometry(point, cache_key, signature)
    if cached is not None:
        return cached

    if num_points <= 1:
        s_coord = coord.new_zeros((num_points,))
        _store_geometry(point, cache_key, signature, s_coord)
        return s_coord

    metric, _, _, _ = manifold_geometry_components(point, k=k_ref)
    assert_index_range("geobias.canonical_arclength.permutation", permutation, num_points)
    coord_c = coord[permutation]
    batch_c = batch[permutation]
    metric_c = metric[permutation].float()

    step = torch.norm(coord_c[1:] - coord_c[:-1], dim=-1)
    local_scale = (0.5 * (metric_c[1:] + metric_c[:-1])).clamp_min(eps)
    step = torch.minimum(step, float(clip_ratio) * local_scale) / local_scale
    same_cloud = batch_c[1:] == batch_c[:-1]
    step = torch.where(same_cloud, step, torch.zeros_like(step))

    s_c = torch.cat([step.new_zeros(1), torch.cumsum(step, dim=0)])
    s_c = _percloud_offsets(s_c, batch_c)

    s_coord = torch.empty_like(s_c)
    assert_index_range(
        "geobias.canonical_arclength.inverse_permutation",
        permutation,
        s_coord.shape[0],
    )
    s_coord[permutation] = s_c
    assert_finite("geobias.canonical_arclength.s_coord", s_coord)
    _store_geometry(point, cache_key, signature, s_coord)
    return s_coord


def _standardize_feature(raw, eps=1.0e-6):
    mean = raw.mean()
    std = raw.std(unbiased=False).clamp_min(eps)
    return (raw - mean) / std


def _batch_standardize_features(raw, batch, eps=1.0e-6):
    if raw.shape[0] == 0:
        return raw
    num_clouds = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_clouds == 0:
        return raw
    assert_index_range("geobias.batch_standardize.batch", batch, num_clouds)
    mean = _scatter_mean(raw, batch, num_clouds)
    centered = raw - mean[batch]
    variance = _scatter_mean(centered.square(), batch, num_clouds)
    return centered / torch.sqrt(variance[batch].clamp_min(eps))


@torch.no_grad()
def geometry_reliability(point, k=(4, 6, 10)):

    k_values = _normalize_k_values(k)
    raw_coord = _coord_source(point)
    signature = (_coord_signature(raw_coord), k_values)
    cache_key = "geobias_reliability_k" + "_".join(str(value) for value in k_values)
    cached = _cached_geometry(point, cache_key, signature)
    if cached is not None:
        return cached

    coord = raw_coord.float()
    batch = _point_batch(point, coord)
    if coord.shape[0] == 0 or len(k_values) == 1:
        reliability = coord.new_ones((coord.shape[0],))
        _store_geometry(point, cache_key, signature, reliability)
        return reliability

    components = [manifold_geometry_components(point, value) for value in k_values]
    disagreement = []


    for component_index in (0, 1, 2):
        normalized = torch.stack(
            [
                _batch_standardize_features(
                    values[component_index].float()[:, None], batch
                ).squeeze(-1)
                for values in components
            ],
            dim=0,
        )
        disagreement.append(normalized.var(dim=0, unbiased=False))

    variance = torch.stack(disagreement, dim=-1).mean(dim=-1)
    uncertainty = (1.0 - torch.exp(-variance)).clamp(0.0, 1.0)
    reliability = (1.0 - uncertainty).clamp(0.0, 1.0)
    assert_finite("geobias.reliability", reliability)
    _store_geometry(point, cache_key, signature, reliability)
    return reliability


def geobias_alibi_slopes(
    num_heads,
    device=None,
    dtype=torch.float32,
    local_fraction=0.5,
):
    slopes = torch.zeros(num_heads, device=device, dtype=dtype)
    local_heads = int(round(float(num_heads) * float(local_fraction)))
    local_heads = max(0, min(int(num_heads), local_heads))
    if local_heads == 0:
        return slopes

    exponents = torch.linspace(
        -8.0,
        -1.0,
        steps=local_heads,
        device=device,
        dtype=torch.float32,
    )
    slopes[:local_heads] = torch.pow(2.0, exponents).to(dtype)
    return slopes


class IntrinsicGeometryDescriptor(nn.Module):
    def __init__(self, k=8, arclength_clip_ratio=3.0):
        super().__init__()
        self.k_values = _normalize_k_values(k)
        self.k_ref = self.k_values[len(self.k_values) // 2]
        self.arclength_clip_ratio = float(arclength_clip_ratio)

    def _components(self, point):
        components = [manifold_geometry_components(point, k=k) for k in self.k_values]
        if len(components) == 1:
            return components[0]
        return tuple(
            torch.stack(values, dim=0).mean(dim=0)
            for values in zip(*components)
        )

    def reliability(self, point):
        return geometry_reliability(point, k=self.k_values)

    def forward(
        self,
        point,
        order=None,
        dtype=torch.float32,
    ):
        coord = _point_coordinates(point)
        batch = _point_batch(point, coord)
        metric, curvature, torsion, confidence = self._components(point)
        s_coord = canonical_arclength(
            point,
            k_ref=self.k_ref,
            clip_ratio=self.arclength_clip_ratio,
        )

        if order is not None:
            assert_index_range("geobias.descriptor.order.coord", order, coord.shape[0])
            assert_index_range("geobias.descriptor.order.metric", order, metric.shape[0])
            coord = coord[order]
            batch = batch[order]
            metric = metric[order]
            curvature = curvature[order]
            torsion = torsion[order]
            confidence = confidence[order]
            s_coord = s_coord[order]

        coord = coord.float()
        confidence = confidence.float().clamp(0.0, 1.0)
        theta_coord = _standardize_feature(curvature.float())
        tau_coord = _standardize_feature(torsion.float())
        intrinsic = torch.stack([s_coord, theta_coord, tau_coord], dim=-1)
        assert_finite("geobias.descriptor.intrinsic", intrinsic)

        features = torch.stack(
            [
                _standardize_feature(metric.float()),
                theta_coord,
                tau_coord,
                confidence * 2.0 - 1.0,
            ],
            dim=-1,
        )
        assert_finite("geobias.descriptor.features", features)
        return intrinsic.to(dtype), features.to(dtype)


class LowRankGeoBias(nn.Module):
    def __init__(
        self,
        num_heads,
        num_frequencies=2,
        descriptor_rank=2,
        k=8,
        head_dim_multiple=8,
        spectral_scale=1.0,
        descriptor_scale=1.0,
        patch_hint=1024,
        stage_index=None,
        geobias_trust_region=0.0,
        geobias_distance_rank=0,
        geobias_distance_gamma_max=0.0,
        geobias_distance_stage_strengths=None,
        geobias_distance_gamma_init=-2.0,
        geobias_distance_projection_init=0.05,
        geobias_uncertainty_budget=0.0,
        geobias_uncertainty_init=-3.0,
    ):
        super().__init__()
        if num_frequencies < 0:
            raise ValueError("num_frequencies must be non-negative")
        if descriptor_rank < 0:
            raise ValueError("descriptor_rank must be non-negative")
        if geobias_distance_rank < 0:
            raise ValueError("geobias_distance_rank must be non-negative")

        self.num_heads = int(num_heads)
        self.num_frequencies = int(num_frequencies)
        self.descriptor_rank = int(descriptor_rank)
        self.head_dim_multiple = int(head_dim_multiple)
        self.spectral_scale = float(spectral_scale)
        self.descriptor_scale = float(descriptor_scale)
        self.descriptor = IntrinsicGeometryDescriptor(k=k)
        self.stage_index = stage_index
        self.geobias_trust_region = max(0.0, float(geobias_trust_region))
        self.distance_rank = int(geobias_distance_rank)
        self.distance_gamma_max = max(0.0, float(geobias_distance_gamma_max))
        self.uncertainty_budget = max(0.0, min(1.0, float(geobias_uncertainty_budget)))
        self.distance_stage_strength = max(
            0.0,
            float(
                _stage_value(
                    geobias_distance_stage_strengths,
                    stage_index,
                    1.0,
                )
            ),
        )
        self.use_distance_basis = (
            self.distance_rank > 0
            and self.distance_gamma_max > 0.0
            and self.distance_stage_strength > 0.0
        )
        self.distance_input_dim = 4
        if (
            self.num_frequencies == 0
            and self.descriptor_rank == 0
            and not self.use_distance_basis
        ):
            raise ValueError("geobias needs at least one active bias channel")

        if self.num_frequencies > 0:
            patch_hint = max(float(patch_hint), 1.0)
            exponents = torch.linspace(
                0.0,
                -math.log2(patch_hint),
                self.num_frequencies,
            )
            base_s = math.pi * torch.pow(2.0, exponents)
            base_angle = math.pi * (
                2.0 ** torch.arange(self.num_frequencies).float()
            )
            axis_base = torch.stack([base_s, base_angle, base_angle], dim=-1)
            self.register_buffer("axis_base", axis_base, persistent=False)
            self.freq_direction = nn.Parameter(
                torch.empty(self.num_heads, self.num_frequencies, 3)
            )
            nn.init.normal_(self.freq_direction, mean=0.0, std=3.0 ** -0.5)
            self.query_spectral_gate = nn.Parameter(
                torch.ones(self.num_heads, self.num_frequencies)
            )
            self.key_spectral_gate = nn.Parameter(
                torch.zeros(self.num_heads, self.num_frequencies)
            )
            if self.geobias_trust_region > 0.0:
                self.spectral_trust_gate = nn.Parameter(torch.zeros(self.num_heads))
            else:
                self.register_parameter("spectral_trust_gate", None)
        else:
            self.register_buffer(
                "axis_base",
                torch.empty(0, 3),
                persistent=False,
            )
            self.register_parameter("freq_direction", None)
            self.register_parameter("query_spectral_gate", None)
            self.register_parameter("key_spectral_gate", None)
            self.register_parameter("spectral_trust_gate", None)

        if self.descriptor_rank > 0:
            self.descriptor_query = nn.Parameter(
                torch.empty(self.num_heads, 4, self.descriptor_rank)
            )
            self.descriptor_key = nn.Parameter(
                torch.zeros(self.num_heads, 4, self.descriptor_rank)
            )
            nn.init.normal_(self.descriptor_query, mean=0.0, std=0.5)
            if self.geobias_trust_region > 0.0:
                self.descriptor_trust_gate = nn.Parameter(torch.zeros(self.num_heads))
            else:
                self.register_parameter("descriptor_trust_gate", None)
        else:
            self.register_parameter("descriptor_query", None)
            self.register_parameter("descriptor_key", None)
            self.register_parameter("descriptor_trust_gate", None)

        if self.use_distance_basis:
            self.distance_projection = nn.Parameter(
                torch.empty(
                    self.num_heads,
                    self.distance_input_dim,
                    self.distance_rank,
                )
            )
            nn.init.normal_(
                self.distance_projection,
                mean=0.0,
                std=float(geobias_distance_projection_init),
            )
            self.raw_distance_gamma = nn.Parameter(
                torch.full(
                    (self.num_heads,),
                    float(geobias_distance_gamma_init),
                )
            )
        else:
            self.register_parameter("distance_projection", None)
            self.register_parameter("raw_distance_gamma", None)

        if self.uncertainty_budget > 0.0:
            self.raw_uncertainty_trust = nn.Parameter(
                torch.full((self.num_heads,), float(geobias_uncertainty_init))
            )
        else:
            self.register_parameter("raw_uncertainty_trust", None)

    @property
    def raw_extra_dim(self):
        dim = 2 * self.num_frequencies + self.descriptor_rank
        if self.use_distance_basis:
            dim += self.distance_rank + 2
        return dim

    def extra_dim(self, base_head_dim):
        raw = self.raw_extra_dim
        if self.head_dim_multiple <= 1:
            return raw
        remainder = (int(base_head_dim) + raw) % self.head_dim_multiple
        padding = (self.head_dim_multiple - remainder) % self.head_dim_multiple
        return raw + padding

    def _trust_multiplier(self, gate, dtype):
        if gate is None or self.geobias_trust_region <= 0.0:
            return None
        multiplier = 1.0 + self.geobias_trust_region * torch.tanh(gate.float())
        multiplier = torch.sqrt(multiplier.clamp_min(1.0e-6))
        return multiplier.to(dtype)[None, :, None]

    def spectral_features(self, intrinsic, dtype=torch.float32):
        if self.num_frequencies == 0:
            empty = intrinsic.new_zeros((intrinsic.shape[0], self.num_heads, 0))
            return empty.to(dtype), empty.to(dtype)

        with _autocast_disabled(intrinsic.device):
            phase = torch.einsum(
                "nd,hfd->nhf",
                intrinsic.float(),
                self.freq_direction.float() * self.axis_base.float()[None],
            )
            assert_finite("geobias.spectral.phase", phase)
            phase = torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi
            assert_finite("geobias.spectral.phase_wrapped", phase)
            paired = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)
            assert_finite("geobias.spectral.paired", paired)
            q_gate = self.query_spectral_gate.float()[None, :, :, None]
            k_gate = self.key_spectral_gate.float()[None, :, :, None]
            q_extra = (paired * q_gate).reshape(
                intrinsic.shape[0],
                self.num_heads,
                2 * self.num_frequencies,
            )
            k_extra = (paired * k_gate).reshape(
                intrinsic.shape[0],
                self.num_heads,
                2 * self.num_frequencies,
            )
            trust = self._trust_multiplier(self.spectral_trust_gate, q_extra.dtype)
            if trust is not None:
                q_extra = q_extra * trust
                k_extra = k_extra * trust
        return (
            (self.spectral_scale * q_extra).to(dtype),
            (self.spectral_scale * k_extra).to(dtype),
        )

    def descriptor_features(self, features, dtype=torch.float32):
        if self.descriptor_rank == 0:
            empty = features.new_zeros((features.shape[0], self.num_heads, 0))
            return empty.to(dtype), empty.to(dtype)

        with _autocast_disabled(features.device):
            q_extra = torch.einsum(
                "nd,hdr->nhr",
                features.float(),
                self.descriptor_query.float(),
            )
            k_extra = torch.einsum(
                "nd,hdr->nhr",
                features.float(),
                self.descriptor_key.float(),
            )
            trust = self._trust_multiplier(self.descriptor_trust_gate, q_extra.dtype)
            if trust is not None:
                q_extra = q_extra * trust
                k_extra = k_extra * trust
            assert_finite("geobias.descriptor.q_extra", q_extra)
            assert_finite("geobias.descriptor.k_extra", k_extra)
        return (
            (self.descriptor_scale * q_extra).to(dtype),
            (self.descriptor_scale * k_extra).to(dtype),
        )

    def distance_features(self, point, order=None, dtype=torch.float32):
        coord = _point_coordinates(point).float()
        batch = _point_batch(point, coord)
        s_coord = canonical_arclength(
            point,
            k_ref=self.descriptor.k_ref,
            clip_ratio=self.descriptor.arclength_clip_ratio,
        ).float()
        coord_feature = _batch_standardize_features(coord, batch)
        s_feature = _batch_standardize_features(s_coord[:, None], batch)
        features = torch.cat([coord_feature, s_feature], dim=-1)
        if order is not None:
            assert_index_range("geobias.distance.order", order, features.shape[0])
            features = features[order]
        assert_finite("geobias.distance.features", features)
        return features.to(dtype)

    def distance_basis_features(self, features, dtype=torch.float32):
        if not self.use_distance_basis:
            empty = features.new_zeros((features.shape[0], self.num_heads, 0))
            return empty.to(dtype), empty.to(dtype)

        with _autocast_disabled(features.device):
            projected = torch.einsum(
                "nd,hdr->nhr",
                features.float(),
                self.distance_projection.float(),
            )
            gamma = (
                self.distance_gamma_max
                * self.distance_stage_strength
                * torch.sigmoid(self.raw_distance_gamma.float())
            )
            sqrt_two_gamma = torch.sqrt((2.0 * gamma).clamp_min(0.0))[None, :, None]
            norm_penalty = gamma[None, :] * projected.square().sum(dim=-1)
            ones = torch.ones_like(norm_penalty[..., None])
            q_extra = torch.cat(
                [
                    sqrt_two_gamma * projected,
                    -norm_penalty[..., None],
                    ones,
                ],
                dim=-1,
            )
            k_extra = torch.cat(
                [
                    sqrt_two_gamma * projected,
                    ones,
                    -norm_penalty[..., None],
                ],
                dim=-1,
            )
            assert_finite("geobias.distance.projected", projected)
            assert_finite("geobias.distance.q_extra", q_extra)
            assert_finite("geobias.distance.k_extra", k_extra)
        return q_extra.to(dtype), k_extra.to(dtype)

    def _reliability_multiplier(self, point, order, dtype):
        if self.uncertainty_budget <= 0.0:
            return None
        reliability = self.descriptor.reliability(point)
        assert_index_range("geobias.reliability.order", order, reliability.shape[0])
        uncertainty = (1.0 - reliability[order]).float()
        rho = self.uncertainty_budget * torch.sigmoid(
            self.raw_uncertainty_trust.float()
        )
        budget = (1.0 - uncertainty[:, None] * rho[None, :]).clamp_min(1.0e-6)
        return torch.sqrt(budget).to(dtype)[..., None]

    def forward(self, point, order, base_head_dim, dtype=torch.float32):
        intrinsic, features = self.descriptor(
            point,
            order=order,
            dtype=torch.float32,
        )
        q_spectral, k_spectral = self.spectral_features(intrinsic, dtype=dtype)
        q_descriptor, k_descriptor = self.descriptor_features(features, dtype=dtype)
        if self.use_distance_basis:
            distance_features = self.distance_features(
                point,
                order=order,
                dtype=torch.float32,
            )
            q_distance, k_distance = self.distance_basis_features(
                distance_features,
                dtype=dtype,
            )
        else:
            empty = features.new_zeros((features.shape[0], self.num_heads, 0))
            q_distance = empty.to(dtype)
            k_distance = empty.to(dtype)
        assert_finite("geobias.q_spectral", q_spectral)
        assert_finite("geobias.k_spectral", k_spectral)
        assert_finite("geobias.q_descriptor", q_descriptor)
        assert_finite("geobias.k_descriptor", k_descriptor)
        assert_finite("geobias.q_distance", q_distance)
        assert_finite("geobias.k_distance", k_distance)

        q_extra = torch.cat([q_spectral, q_descriptor, q_distance], dim=-1)
        k_extra = torch.cat([k_spectral, k_descriptor, k_distance], dim=-1)
        reliability_multiplier = self._reliability_multiplier(point, order, dtype)
        if reliability_multiplier is not None:
            q_extra = q_extra * reliability_multiplier
            k_extra = k_extra * reliability_multiplier
        padding = self.extra_dim(base_head_dim) - self.raw_extra_dim
        if padding > 0:
            zeros = q_extra.new_zeros(q_extra.shape[:-1] + (padding,))
            q_extra = torch.cat([q_extra, zeros], dim=-1)
            k_extra = torch.cat([k_extra, zeros], dim=-1)

        v_extra = q_extra.new_zeros(q_extra.shape)
        assert_finite("geobias.q_extra", q_extra)
        assert_finite("geobias.k_extra", k_extra)
        return q_extra.to(dtype), k_extra.to(dtype), v_extra.to(dtype)
