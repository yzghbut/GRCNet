import sys
import inspect
import math
from collections import OrderedDict
from functools import partial

import flash_attn
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch_scatter

from addict import Dict
from timm.layers import DropPath

from .geobias import LowRankGeoBias, geobias_alibi_slopes
from .scene_query import GlobalSceneQuery
from .serialization import encode
from utils.debug_indices import assert_index_range


def _flash_varlen_supports_alibi():
    try:
        parameters = inspect.signature(
            flash_attn.flash_attn_varlen_qkvpacked_func
        ).parameters
    except (TypeError, ValueError):
        return False
    return "alibi_slopes" in parameters


def _stage_option(value, index, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return value[min(int(index), len(value) - 1)]
    return value


def _cpe_hidden_channels(channels, ratio, channel_multiple):
    hidden = max(1, int(math.ceil(float(channels) * float(ratio))))
    channel_multiple = int(channel_multiple)
    if channel_multiple > 1:
        hidden = int(math.ceil(hidden / channel_multiple) * channel_multiple)
    return min(int(channels), hidden)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)

@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):

        self["order"] = order
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():


            assert {"grid_size", "coord"}.issubset(self.keys())

            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:

            depth = int(self.grid_coord.max() + 1).bit_length()
        self["serialized_depth"] = depth

        assert depth * 3 + len(self.offset).bit_length() <= 63


        assert depth <= 16


        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )
        self["serialized_order_canonical"] = order[0].clone()

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):

        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():


            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    @torch.no_grad()
    def get_padding_and_inverse(self, patch_size):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        padding_signature_key = "padding_signature"
        offset = self.offset
        padding_signature = (
            int(patch_size),
            tuple(offset.shape),
            str(offset.device),
            offset.data_ptr(),
            offset._version,
        )
        if (
            pad_key not in self.keys()
            or unpad_key not in self.keys()
            or cu_seqlens_key not in self.keys()
            or self.get(padding_signature_key) != padding_signature
        ):
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + patch_size - 1,
                    patch_size,
                    rounding_mode="trunc",
                )
                * patch_size
            )

            mask_pad = bincount > patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - patch_size
                        + (bincount[i] % patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * patch_size
                        + (bincount[i] % patch_size) : _offset_pad[i + 1]
                        - patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            self[pad_key] = pad
            self[unpad_key] = unpad
            self[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
            self[padding_signature_key] = padding_signature
        return self[pad_key], self[unpad_key], self[cu_seqlens_key]


class PointModule(nn.Module):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):


    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():

            if isinstance(module, PointModule):
                input = module(input)

            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)

            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class GeoBiasAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        stage_index=None,
        geobias_num_frequencies=2,
        geobias_descriptor_rank=2,
        geobias_k=8,
        geobias_head_dim_multiple=8,
        geobias_bias_scale=None,
        geobias_trust_region=0.0,
        geobias_distance_rank=0,
        geobias_distance_gamma_max=0.0,
        geobias_distance_stage_strengths=None,
        geobias_distance_gamma_init=-2.0,
        geobias_distance_projection_init=0.05,
        geobias_uncertainty_budget=0.0,
        geobias_uncertainty_init=-3.0,
        enable_alibi=False,
        alibi_local_fraction=0.5,
        enable_geobias=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index

        self.patch_size = patch_size
        self.attn_drop = attn_drop

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.enable_geobias = bool(enable_geobias)
        self.geobias = None
        if self.enable_geobias:
            self.geobias = LowRankGeoBias(
                num_heads=num_heads,
                num_frequencies=geobias_num_frequencies,
                descriptor_rank=geobias_descriptor_rank,
                k=geobias_k,
                head_dim_multiple=geobias_head_dim_multiple,
                patch_hint=patch_size,
                stage_index=stage_index,
                geobias_trust_region=geobias_trust_region,
                geobias_distance_rank=geobias_distance_rank,
                geobias_distance_gamma_max=geobias_distance_gamma_max,
                geobias_distance_stage_strengths=geobias_distance_stage_strengths,
                geobias_distance_gamma_init=geobias_distance_gamma_init,
                geobias_distance_projection_init=geobias_distance_projection_init,
                geobias_uncertainty_budget=geobias_uncertainty_budget,
                geobias_uncertainty_init=geobias_uncertainty_init,
            )
        self.geobias_bias_scale = (
            float(geobias_bias_scale)
            if geobias_bias_scale is not None
            else float(self.scale) ** -0.5
        )
        self.enable_alibi = bool(enable_alibi) and _flash_varlen_supports_alibi()
        self.alibi_local_fraction = alibi_local_fraction

    def _flash_attention(self, qkv, cu_seqlens):
        kwargs = {}
        if self.enable_alibi:
            kwargs["alibi_slopes"] = geobias_alibi_slopes(
                self.num_heads,
                device=qkv.device,
                dtype=torch.float32,
                local_fraction=self.alibi_local_fraction,
            )
        return flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.attn_drop if self.training else 0,
            softmax_scale=self.scale,
            **kwargs,
        )

    def forward(self, point):

        H = self.num_heads
        C = self.channels
        head_dim = C // H

        pad, unpad, cu_seqlens = point.get_padding_and_inverse(self.patch_size)

        serialized_order = point.serialized_order[self.order_index]
        serialized_inverse = point.serialized_inverse[self.order_index]
        assert_index_range("standalone_attention.pad", pad, serialized_order.shape[0])
        assert_index_range(
            "standalone_attention.serialized_inverse",
            serialized_inverse,
            unpad.shape[0],
        )
        order = serialized_order[pad]
        inverse = unpad[serialized_inverse]
        assert_index_range("standalone_attention.order.feat", order, point.feat.shape[0])


        qkv = self.qkv(point.feat)
        qkv = qkv[order]

        q, k, v = qkv.half().chunk(3, dim=-1)
        q = q.reshape(-1, H, head_dim).transpose(0, 1)[None]
        k = k.reshape(-1, H, head_dim).transpose(0, 1)[None]

        q_flash = q.squeeze(0).transpose(0, 1)
        k_flash = k.squeeze(0).transpose(0, 1)
        v_flash = v.reshape(-1, H, head_dim)
        if self.geobias is not None:
            q_extra, k_extra, v_extra = self.geobias(
                point,
                order=order,
                base_head_dim=head_dim,
                dtype=q_flash.dtype,
            )
            q_extra = q_extra * self.geobias_bias_scale
            k_extra = k_extra * self.geobias_bias_scale
            q_flash = torch.cat([q_flash, q_extra], dim=-1)
            k_flash = torch.cat([k_flash, k_extra], dim=-1)
            v_flash = torch.cat([v_flash, v_extra], dim=-1)

        qkv_input = torch.stack(
            [q_flash, k_flash, v_flash],
            dim=1,
        ).to(v.dtype)

        feat = self._flash_attention(qkv_input, cu_seqlens)
        feat = feat[:, :, :head_dim]
        feat = feat.reshape(-1, C)

        feat = feat.to(qkv.dtype)
        assert_index_range("standalone_attention.inverse", inverse, feat.shape[0])
        feat = feat[inverse]


        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
        re_serialization=False,
        serialization_order="z",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

        self.re_serialization = re_serialization
        self.serialization_order = serialization_order

    def forward(self, point: Point):
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[gird_coord] or [coord, grid_size] should be include in the Point"
            )
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)

        _, indices = torch.sort(cluster)

        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

        assert_index_range(
            "standalone_grid_pooling.indices.feat",
            indices,
            point.feat.shape[0],
        )
        assert_index_range(
            "standalone_grid_pooling.indices.coord",
            indices,
            point.coord.shape[0],
        )
        head_indices = indices[idx_ptr[:-1]]
        assert_index_range(
            "standalone_grid_pooling.head_indices.batch",
            head_indices,
            point.batch.shape[0],
        )
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        if "origin_coord" in point.keys():
            assert_index_range(
                "standalone_grid_pooling.indices.origin_coord",
                indices,
                point.origin_coord.shape[0],
            )
            point_dict["origin_coord"] = torch_scatter.segment_csr(
                point.origin_coord[indices], idx_ptr, reduce="mean"
            )
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "split" in point.keys():
            point_dict["split"] = point.split
        if "color" in point.keys():
            assert_index_range(
                "standalone_grid_pooling.indices.color",
                indices,
                point.color.shape[0],
            )
            point_dict["color"] = torch_scatter.segment_csr(
                point.color[indices], idx_ptr, reduce="mean"
            )
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride
        if "mask" in point.keys():
            assert_index_range(
                "standalone_grid_pooling.indices.mask",
                indices,
                point.mask.shape[0],
            )
            point_dict["mask"] = torch_scatter.segment_csr(
                point.mask[indices].float(), idx_ptr, reduce="mean"
            ) > 0.5

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)

        if self.re_serialization:
            point.serialization(order=self.serialization_order, shuffle_orders=self.shuffle_orders and self.training)
        point.sparsify()
        return point


class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,
        reliability_gate=False,
        reliability_floor=0.5,
        reliability_gate_init=-3.0,
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if not 0.0 <= float(reliability_floor) <= 1.0:
            raise ValueError("reliability_floor must be in [0, 1]")
        self.reliability_gate = bool(reliability_gate)
        self.reliability_floor = float(reliability_floor)
        if self.reliability_gate:


            self.reliability_gate_logit = nn.Parameter(
                torch.tensor(float(reliability_gate_init))
            )
        else:
            self.register_parameter("reliability_gate_logit", None)

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat = point.feat

        parent = self.proj_skip(parent)
        projected = self.proj(point)
        assert_index_range(
            "standalone_grid_unpooling.inverse",
            inverse,
            projected.feat.shape[0],
        )
        projected_feat = projected.feat[inverse]
        if self.reliability_gate and "geometry_reliability" in point.keys():
            reliability = point.geometry_reliability[inverse].float().clamp(0.0, 1.0)
            suppression = torch.sigmoid(self.reliability_gate_logit).to(
                dtype=projected_feat.dtype
            )
            gate = 1.0 - suppression * (1.0 - reliability)
            gate = self.reliability_floor + (1.0 - self.reliability_floor) * gate
            projected_feat = projected_feat * gate[:, None].to(projected_feat.dtype)
        parent.feat = parent.feat + projected_feat
        if "scene_query_semantic_logits" in point.keys():
            semantic_logits = point.scene_query_semantic_logits
            assert_index_range(
                "standalone_grid_unpooling.scene_query_semantic_logits.inverse",
                inverse,
                semantic_logits.shape[0],
            )
            parent.scene_query_semantic_logits = semantic_logits[inverse]
        if "geometry_reliability" in point.keys():
            parent.geometry_reliability = point.geometry_reliability[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
            parent["unpooling_inverse"] = inverse
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels


        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        cpe_mode="full",
        cpe_ratio=0.125,
        cpe_channel_multiple=8,
        enable_conv=True,
        enable_attn=True,
        stage_index=None,
        geobias_num_frequencies=2,
        geobias_descriptor_rank=2,
        geobias_k=8,
        geobias_head_dim_multiple=8,
        geobias_bias_scale=None,
        geobias_trust_region=0.0,
        geobias_distance_rank=0,
        geobias_distance_gamma_max=0.0,
        geobias_distance_stage_strengths=None,
        geobias_distance_gamma_init=-2.0,
        geobias_distance_projection_init=0.05,
        geobias_uncertainty_budget=0.0,
        geobias_uncertainty_init=-3.0,
        enable_alibi=False,
        alibi_local_fraction=0.5,
        enable_geobias=True,
        scene_query=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.enable_conv = enable_conv
        self.enable_attn = enable_attn
        self.scene_query = scene_query
        self.has_block_mlp = not (
            self.scene_query is not None and self.scene_query.replace_ffn
        )

        if self.enable_conv:
            cpe_mode = str(cpe_mode).lower()
            if cpe_mode == "full":
                self.conv = PointSequential(
                    spconv.SubMConv3d(
                        channels,
                        channels,
                        kernel_size=3,
                        bias=True,
                        indice_key=cpe_indice_key,
                    ),
                    nn.Linear(channels, channels),
                    norm_layer(channels),
                )
            elif cpe_mode == "bottleneck":
                hidden_channels = _cpe_hidden_channels(
                    channels,
                    cpe_ratio,
                    cpe_channel_multiple,
                )
                bottleneck_key = (
                    None
                    if cpe_indice_key is None
                    else f"{cpe_indice_key}_bottleneck"
                )
                self.conv = PointSequential(
                    nn.Linear(channels, hidden_channels),
                    spconv.SubMConv3d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        bias=True,
                        indice_key=bottleneck_key,
                    ),
                    nn.Linear(hidden_channels, channels),
                    norm_layer(channels),
                )
            else:
                raise ValueError(f"Unsupported CPE mode: {cpe_mode}")
        else:
            self.norm0 = PointSequential(
                norm_layer(channels),
            )


        if self.enable_attn:
            self.norm1 = PointSequential(norm_layer(channels))
            self.attn = None if self.scene_query is not None else GeoBiasAttention(
                channels=channels,
                patch_size=patch_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                order_index=order_index,
                stage_index=stage_index,
                geobias_num_frequencies=geobias_num_frequencies,
                geobias_descriptor_rank=geobias_descriptor_rank,
                geobias_k=geobias_k,
                geobias_head_dim_multiple=geobias_head_dim_multiple,
                geobias_bias_scale=geobias_bias_scale,
                geobias_trust_region=geobias_trust_region,
                geobias_distance_rank=geobias_distance_rank,
                geobias_distance_gamma_max=geobias_distance_gamma_max,
                geobias_distance_stage_strengths=geobias_distance_stage_strengths,
                geobias_distance_gamma_init=geobias_distance_gamma_init,
                geobias_distance_projection_init=geobias_distance_projection_init,
                enable_alibi=enable_alibi,
                alibi_local_fraction=alibi_local_fraction,
                enable_geobias=enable_geobias,
            )
            if self.has_block_mlp:
                self.norm2 = PointSequential(norm_layer(channels))
                self.mlp = PointSequential(
                    MLP(
                        in_channels=channels,
                        hidden_channels=int(channels * mlp_ratio),
                        out_channels=channels,
                        act_layer=act_layer,
                        drop=proj_drop,
                    )
                )
            self.drop_path = PointSequential(
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

    def forward(self, point: Point):
        if self.enable_conv:
            shortcut = point.feat
            point = self.conv(point)
            point.feat = shortcut + point.feat
        else:
            point = self.norm0(point)

        if self.enable_attn:
            shortcut = point.feat
            if self.pre_norm:
                point = self.norm1(point)
            if self.scene_query is not None:
                normalized = point.feat
                point = self.scene_query(point)
                point.feat = shortcut + self.drop_path(point.feat - normalized)
            else:
                point = self.drop_path(self.attn(point))
                point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm1(point)

            if self.has_block_mlp:
                shortcut = point.feat
                if self.pre_norm:
                    point = self.norm2(point)
                point = self.drop_path(self.mlp(point))
                point.feat = shortcut + point.feat
                if not self.pre_norm:
                    point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class LitePT(PointModule):
    def __init__(
        self,
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_cpe_mode=("full", "full", "full", "full", "full"),
        enc_attn=(False, False, False, True, True),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_cpe_mode=("full", "full", "full", "full"),
        dec_attn=(False, False, False, False),
        dec_reliability_gate=False,
        dec_reliability_floor=0.5,
        dec_reliability_gate_init=-3.0,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
        cpe_ratio=0.125,
        cpe_channel_multiple=8,
        enable_geobias=True,
        scene_query_num_tokens=0,
        scene_query_rank=16,
        scene_query_coord_frequencies=4,
        scene_query_num_classes=0,
        scene_query_assignment_mode="softmax",
        scene_query_sinkhorn_iterations=5,
        scene_query_sinkhorn_epsilon=0.1,
        scene_query_sinkhorn_mass_relaxation=0.5,
        scene_query_layer_scale_init=1.0e-3,
        scene_query_use_geometry_reliability=False,
        scene_query_geometry_k=(4, 6, 10),
        scene_query_shared_basis=False,
        scene_query_replace_ffn=False,
        scene_query_ffn_rank=128,
        scene_query_output_adapter_rank=0,
        scene_query_local_reliability_floor=0.5,
        scene_query_global_reliability_floor=0.5,
        geobias_num_frequencies=2,
        geobias_descriptor_rank=2,
        geobias_k=8,
        geobias_head_dim_multiple=8,
        geobias_bias_scale=None,
        geobias_trust_region=0.0,
        geobias_distance_rank=0,
        geobias_distance_gamma_max=0.0,
        geobias_distance_stage_strengths=None,
        geobias_distance_gamma_init=-2.0,
        geobias_distance_projection_init=0.05,
        geobias_uncertainty_budget=0.0,
        geobias_uncertainty_init=-3.0,
        enable_alibi=False,
        alibi_local_fraction=0.5,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders

        self.enc_conv = enc_conv
        self.enc_attn = enc_attn
        self.dec_conv = dec_conv
        self.dec_attn = dec_attn
        self.enable_geobias = bool(enable_geobias)
        scene_query_module = (
            GlobalSceneQuery(
                channels=enc_channels[-1],
                num_tokens=scene_query_num_tokens,
                query_rank=scene_query_rank,
                num_coord_frequencies=scene_query_coord_frequencies,
                num_classes=scene_query_num_classes,
                assignment_mode=scene_query_assignment_mode,
                sinkhorn_iterations=scene_query_sinkhorn_iterations,
                sinkhorn_epsilon=scene_query_sinkhorn_epsilon,
                sinkhorn_mass_relaxation=scene_query_sinkhorn_mass_relaxation,
                layer_scale_init=scene_query_layer_scale_init,
                use_geometry_reliability=scene_query_use_geometry_reliability,
                geometry_k=scene_query_geometry_k,
                shared_basis=scene_query_shared_basis,
                replace_ffn=scene_query_replace_ffn,
                ffn_rank=scene_query_ffn_rank,
                output_adapter_rank=scene_query_output_adapter_rank,
                local_reliability_floor=scene_query_local_reliability_floor,
                global_reliability_floor=scene_query_global_reliability_floor,
                proj_drop=proj_drop,
            )
            if int(scene_query_num_tokens) > 0
            else None
        )

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1


        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm


        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )


        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        re_serialization=enc_attn[s],
                        serialization_order=self.order
                    ),

                    name="down",
                )
            for i in range(enc_depths[s]):
                block_scene_query = (
                    scene_query_module
                    if s == self.num_stages - 1 and i == enc_depths[s] - 1
                    else None
                )
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        cpe_mode=_stage_option(enc_cpe_mode, s, "full"),
                        cpe_ratio=cpe_ratio,
                        cpe_channel_multiple=cpe_channel_multiple,
                        enable_conv=enc_conv[s],
                        enable_attn=enc_attn[s],
                        stage_index=s,
                        geobias_num_frequencies=geobias_num_frequencies,
                        geobias_descriptor_rank=geobias_descriptor_rank,
                        geobias_k=geobias_k,
                        geobias_head_dim_multiple=geobias_head_dim_multiple,
                        geobias_bias_scale=geobias_bias_scale,
                        geobias_trust_region=geobias_trust_region,
                        geobias_distance_rank=geobias_distance_rank,
                        geobias_distance_gamma_max=geobias_distance_gamma_max,
                        geobias_distance_stage_strengths=geobias_distance_stage_strengths,
                         geobias_distance_gamma_init=geobias_distance_gamma_init,
                         geobias_distance_projection_init=geobias_distance_projection_init,
                         geobias_uncertainty_budget=geobias_uncertainty_budget,
                         geobias_uncertainty_init=geobias_uncertainty_init,
                         enable_alibi=enable_alibi,
                        alibi_local_fraction=alibi_local_fraction,
                        enable_geobias=self.enable_geobias,
                        scene_query=block_scene_query,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")


        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        reliability_gate=dec_reliability_gate,
                        reliability_floor=dec_reliability_floor,
                        reliability_gate_init=dec_reliability_gate_init,
                    ),

                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            cpe_mode=_stage_option(dec_cpe_mode, s, "full"),
                            cpe_ratio=cpe_ratio,
                            cpe_channel_multiple=cpe_channel_multiple,
                            enable_conv=dec_conv[s],
                            enable_attn=dec_attn[s],
                            stage_index=s,
                            geobias_num_frequencies=geobias_num_frequencies,
                            geobias_descriptor_rank=geobias_descriptor_rank,
                            geobias_k=geobias_k,
                            geobias_head_dim_multiple=geobias_head_dim_multiple,
                            geobias_bias_scale=geobias_bias_scale,
                            geobias_trust_region=geobias_trust_region,
                            geobias_distance_rank=geobias_distance_rank,
                            geobias_distance_gamma_max=geobias_distance_gamma_max,
                            geobias_distance_stage_strengths=geobias_distance_stage_strengths,
                             geobias_distance_gamma_init=geobias_distance_gamma_init,
                             geobias_distance_projection_init=geobias_distance_projection_init,
                             geobias_uncertainty_budget=geobias_uncertainty_budget,
                             geobias_uncertainty_init=geobias_uncertainty_init,
                             enable_alibi=enable_alibi,
                            alibi_local_fraction=alibi_local_fraction,
                            enable_geobias=self.enable_geobias,
                            scene_query=None,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):

        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders and self.training)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)

        if not self.enc_mode:
            point = self.dec(point)

        return point
