import os
import numpy as np
from collections.abc import Sequence
import pickle

from .builder import DATASETS
from .defaults import DefaultDataset
from .polar_mix import polar_mix


@DATASETS.register_module()
class NuScenesDataset(DefaultDataset):
    def __init__(self, sweeps=10, ignore_index=-1, raw_data_root=None,
                 polarmix=None, **kwargs):
        self.polarmix = dict(polarmix) if polarmix is not None else None
        if self.polarmix is not None:
            if kwargs.get("split", "train") != "train" or kwargs.get("test_mode", False):
                raise ValueError("PolarMix is allowed only on the training split")
        self.sweeps = sweeps
        self.ignore_index = ignore_index


        self.raw_data_root = raw_data_root
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def prepare_train_data(self, idx):
        if self.polarmix is None:
            return super().prepare_train_data(idx)
        source = self.get_data(idx)
        donor = self.get_data(int(np.random.randint(len(self.data_list))))
        return self.transform(polar_mix(source, donor, **self.polarmix))

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        raw_root = self.raw_data_root or os.path.join(self.data_root, "raw")
        lidar_path = os.path.join(raw_root, data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255

        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                raw_root, data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):

        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map
