from utils.registry import Registry

DATASETS = Registry("datasets")


def build_dataset(cfg):

    return DATASETS.build(cfg)
