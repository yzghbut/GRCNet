def require_assets(data_dict, required, data_path, name=None):
    missing = [key for key in required if key not in data_dict]
    if not missing:
        return data_dict

    available = sorted(key for key in data_dict.keys() if not key.startswith("_"))
    sample = f"'{name}'" if name is not None else "<unknown>"
    raise KeyError(
        f"ScanNet sample {sample} is missing required assets {missing} at "
        f"{data_path}. available assets: {available}. "
        "Check that preprocessing produced coord.npy/color.npy/normal.npy and "
        "that data_root points to the processed ScanNet directory."
    )
