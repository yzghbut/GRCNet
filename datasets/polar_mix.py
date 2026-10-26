# Mix only training frames and preserve point-label alignment.
import numpy as np


def polar_mix(source, donor, swap_prob=0.5, paste_prob=1.0,
              object_classes=tuple(range(8)), rng=None, max_points=None,
              paste_rotation_mode="three_copies"):

    rng = np.random if rng is None else rng
    keys = ("coord", "strength", "segment")
    for cloud in (source, donor):
        n = len(cloud["coord"])
        if any(len(cloud[key]) != n for key in keys):
            raise ValueError("PolarMix point features and labels must align")
    if not all(np.isfinite(p) and 0 <= p <= 1 for p in (swap_prob, paste_prob)):
        raise ValueError("PolarMix probabilities must be finite and in [0, 1]")
    if max_points is not None:
        if not isinstance(max_points, (int, np.integer)) or max_points <= 0:
            raise ValueError("PolarMix max_points must be a positive integer")
        if len(source["coord"]) > max_points:
            raise ValueError("Original frame exceeds PolarMix max_points; no points were dropped")
    if paste_rotation_mode not in ("three_copies", "nuscenes_two_copies"):
        raise ValueError("Unknown PolarMix paste_rotation_mode")
    chunks = {key: [source[key]] for key in keys}
    point_count = len(source["coord"])
    if rng.random() < swap_prob:
        alpha = rng.uniform(-np.pi, 0)
        masks = []
        for cloud in (source, donor):
            yaw = -np.arctan2(cloud["coord"][:, 1], cloud["coord"][:, 0])
            masks.append((yaw > alpha) & (yaw < alpha + np.pi))
        swapped_count = int((~masks[0]).sum() + masks[1].sum())
        if max_points is None or swapped_count <= max_points:
            for key in keys:
                chunks[key] = [source[key][~masks[0]], donor[key][masks[1]]]
            point_count = swapped_count
    if rng.random() < paste_prob:
        mask = np.isin(donor["segment"], object_classes)
        xyz = donor["coord"][mask]
        if paste_rotation_mode == "nuscenes_two_copies":
            angles = (0.0, float(rng.choice((-np.pi / 2, np.pi / 2))))
        else:
            angles = (0.0, rng.uniform(0, 2*np.pi/3), rng.uniform(2*np.pi/3, 4*np.pi/3))
        for angle in angles:


            if max_points is not None and point_count + len(xyz) > max_points:
                continue
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=xyz.dtype)
            chunks["coord"].append(xyz @ rotation)
            chunks["strength"].append(donor["strength"][mask])
            chunks["segment"].append(donor["segment"][mask])
            point_count += len(xyz)
    result = dict(source)
    result.update({key: np.concatenate(chunks[key], axis=0) for key in keys})
    return result
