import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from utils.logger import get_root_logger
from utils.cache import shared_dict
from utils.data_validation import require_assets
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)


@DATASETS.register_module()
class ScanNetDataset(DefaultDataset):
    REQUIRED_ASSETS = ("coord", "color", "normal")
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        strict_data=True,
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        self.strict_data = strict_data
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        if not self.strict_data:
            data_list = self._filter_valid_data_list(data_list)
        if len(data_list) == 0:
            raise RuntimeError(
                f"No valid ScanNet samples found under {self.data_root}/{self.split}. "
                f"Each sample directory must contain {self.REQUIRED_ASSETS}."
            )
        return data_list

    def _filter_valid_data_list(self, data_list):
        logger = get_root_logger()
        valid_data_list = []
        skipped = []
        for data_path in data_list:
            if self._has_required_assets(data_path):
                valid_data_list.append(data_path)
            else:
                skipped.append(data_path)
        if skipped:
            preview = ", ".join(skipped[:5])
            suffix = "" if len(skipped) <= 5 else f", ... ({len(skipped)} total)"
            logger.warning(
                f"Skipped ScanNet samples with missing assets: {preview}{suffix}"
            )
        return valid_data_list

    def _has_required_assets(self, data_path):
        if not os.path.isdir(data_path):
            return False
        assets = {
            os.path.splitext(asset)[0]
            for asset in os.listdir(data_path)
            if asset.endswith(".npy")
        }
        return all(key in assets for key in self.REQUIRED_ASSETS)

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["split"] = split
        require_assets(
            data_dict,
            required=self.REQUIRED_ASSETS,
            data_path=data_path,
            name=name,
        )
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index
        return data_dict


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)
