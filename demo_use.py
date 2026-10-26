from collections import OrderedDict

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from datasets.transform import Compose
from litept.model import LitePT

if __name__ == "__main__":


    model = LitePT()
    model.cuda()


    ckpt_path = hf_hub_download(
        repo_id="prs-eth/LitePT",
        filename="nuscenes-semseg-litept-small-v1m1/model/model_best.pth",
        repo_type="model"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    weight = OrderedDict()
    prefix = "module.backbone."
    for key, value in ckpt["state_dict"].items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            weight[new_key] = value
    model.load_state_dict(weight, strict=True)
    model.eval()


    lidar_path = hf_hub_download(
            repo_id="prs-eth/LitePT_demo",
            filename=f"outdoor_sample1.bin",
            repo_type="dataset",
            revision="main",
        )
    points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
    coord = points[:, :3]
    strength = points[:, 3].reshape([-1, 1]) / 255
    point = dict(
        coord=coord,
        strength=strength,
        )
    print("Number of points: ", coord.shape[0])


    data_config = [
        dict(
            type="GridSample",
            grid_size=0.05,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "inverse"),
            feat_keys=("coord", "strength"),
        ),
    ]
    transform = Compose(data_config)
    point = transform(point)


    with torch.no_grad():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)

        point = model(point)

        print("Downsampled output feature: ", point.feat.shape)

        dense_feat = point.feat[point.inverse]
        print("Dense output feature: ", dense_feat.shape)
