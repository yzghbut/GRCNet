def build_loader_runtime_options(cfg, num_workers):
    if cfg is None:
        cfg = object()
    options = {
        "pin_memory": getattr(cfg, "pin_memory", True),
        "persistent_workers": (
            bool(getattr(cfg, "persistent_workers", True)) and num_workers > 0
        ),
    }
    prefetch_factor = getattr(cfg, "prefetch_factor", None)
    if num_workers > 0 and prefetch_factor is not None:
        options["prefetch_factor"] = prefetch_factor
    timeout = getattr(cfg, "dataloader_timeout", 0)
    if num_workers > 0 and timeout:
        options["timeout"] = timeout
    return options
