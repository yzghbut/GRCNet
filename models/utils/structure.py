import torch
import torch.nn as nn
import spconv.pytorch as spconv

from addict import Dict
from typing import List


from models.utils.serialization import encode
from models.utils import (
    offset2batch,
    batch2offset,
    offset2bincount,
    bincount2offset,
)


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
