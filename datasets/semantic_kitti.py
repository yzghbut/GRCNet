import os
import numpy as np
from collections.abc import Sequence

from .builder import DATASETS
from .defaults import DefaultDataset
from .polar_mix import polar_mix


@DATASETS.register_module()
class SemanticKITTIDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, polarmix=None, **kwargs):
        self.polarmix = dict(polarmix) if polarmix is not None else None
        if self.polarmix is not None:
            if kwargs.get("split", "train") != "train" or kwargs.get("test_mode", False):
                raise ValueError("PolarMix is allowed only on the training split")
            for key in ("swap_prob", "paste_prob"):
                value = self.polarmix.get(key, 0.5 if key == "swap_prob" else 1.0)
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid PolarMix {key}: {value}")
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv()
        super().__init__(ignore_index=ignore_index, **kwargs)

    def prepare_train_data(self, idx):
        if self.polarmix is None:
            return super().prepare_train_data(idx)
        source = self.get_data(idx)
        donor = self.get_data(np.random.randint(len(self.data_list)))
        return self.transform(polar_mix(source, donor, **self.polarmix))

    def _sequence_root(self):

        nested = os.path.join(self.data_root, "dataset", "sequences")
        flat = os.path.join(self.data_root, "sequences")
        if os.path.isdir(nested):
            return nested
        if os.path.isdir(flat):
            return flat
        raise FileNotFoundError(
            "SemanticKITTI sequences directory not found under "
            f"{self.data_root!r}; checked {nested!r} and {flat!r}"
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            splits = [self.split]
        elif isinstance(self.split, Sequence):
            splits = list(self.split)
        else:
            raise NotImplementedError

        data_list = []
        sequence_root = self._sequence_root()
        for split in splits:
            sequences = self.get_sequences(split)
            for sequence in sequences:
                velodyne_dir = os.path.join(sequence_root, sequence, "velodyne")
                if not os.path.isdir(velodyne_dir):
                    continue
                for file_name in sorted(os.listdir(velodyne_dir)):
                    if file_name.endswith(".bin"):
                        data_list.append((sequence, file_name[:-4]))
        return data_list

    @staticmethod
    def get_sequences(split):
        if split == "train":
            return [f"{i:02d}" for i in range(11) if i != 8]
        if split == "val":
            return ["08"]
        if split == "test":
            return [f"{i:02d}" for i in range(11, 22)]
        raise ValueError(f"Unsupported SemanticKITTI split: {split}")

    def get_data(self, idx):
        sequence, frame = self.data_list[idx % len(self.data_list)]
        sequence_dir = os.path.join(self._sequence_root(), sequence)

        lidar_path = os.path.join(sequence_dir, "velodyne", f"{frame}.bin")
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        coord = points[:, :3]
        strength = points[:, 3:4].astype(np.float32)

        label_path = os.path.join(sequence_dir, "labels", f"{frame}.label")
        if os.path.isfile(label_path):
            raw_label = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
            raw_label = raw_label & 0xFFFF
            segment = np.vectorize(self.learning_map.__getitem__)(raw_label).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index

        return dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )

    def get_data_name(self, idx):
        sequence, frame = self.data_list[idx % len(self.data_list)]
        return f"{sequence}_{frame}"

    @staticmethod
    def get_learning_map(ignore_index):
        return {
            0: ignore_index,
            1: ignore_index,
            10: 0,
            11: 1,
            13: 4,
            15: 2,
            16: 4,
            18: 3,
            20: 4,
            30: 5,
            31: 6,
            32: 7,
            40: 8,
            44: 9,
            48: 10,
            49: 11,
            50: 12,
            51: 13,
            52: ignore_index,
            60: 8,
            70: 14,
            71: 15,
            72: 16,
            80: 17,
            81: 18,
            99: ignore_index,
            252: 0,
            253: 6,
            254: 5,
            255: 7,
            256: 4,
            257: 4,
            258: 3,
            259: 4,
        }
    @staticmethod
    def get_learning_map_inv():
        return {
            0: 10,
            1: 11,
            2: 15,
            3: 18,
            4: 20,
            5: 30,
            6: 31,
            7: 32,
            8: 40,
            9: 44,
            10: 48,
            11: 49,
            12: 50,
            13: 51,
            14: 70,
            15: 71,
            16: 72,
            17: 80,
            18: 81,
        }
