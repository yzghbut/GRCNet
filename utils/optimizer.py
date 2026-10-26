# Specific parameter groups take precedence over broader keyword matches.
import torch
from utils.logger import get_root_logger
from utils.registry import Registry

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")


def _rule_value(rule, name, default=None):
    if isinstance(rule, dict):
        return rule.get(name, default)
    return getattr(rule, name, default)


def _rule_keywords(rule):
    keywords = _rule_value(rule, "keywords")
    if keywords is None:
        keywords = _rule_value(rule, "keyword")
    if isinstance(keywords, str):
        keywords = (keywords,)
    else:
        try:
            keywords = tuple(keywords)
        except TypeError:
            raise ValueError(
                "optimizer parameter rule needs keyword or keywords"
            ) from None
    if not keywords or not all(
        isinstance(keyword, str) and keyword for keyword in keywords
    ):
        raise ValueError("optimizer parameter rule needs keyword or keywords")
    return keywords


def build_parameter_groups(model, base_lr, param_dicts):
    groups = [dict(names=[], params=[], lr=base_lr)]
    rule_keywords = []

    for rule in param_dicts:
        group = dict(names=[], params=[])
        for option in ("lr", "momentum", "weight_decay"):
            value = _rule_value(rule, option)
            if value is not None:
                group[option] = value
        groups.append(group)
        rule_keywords.append(_rule_keywords(rule))

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        owner = 0
        for index, keywords in enumerate(rule_keywords, start=1):
            if any(keyword in name for keyword in keywords):
                owner = index
                break
        groups[owner]["names"].append(name)
        groups[owner]["params"].append(parameter)

    for keywords, group in zip(rule_keywords, groups[1:]):
        if not group["params"]:
            raise ValueError(
                f"optimizer parameter rule {keywords} matched no trainable parameters"
            )
    return groups


def build_optimizer(cfg, model, param_dicts=None):
    optimizer_cfg = cfg.copy()
    if param_dicts is None:
        parameter_groups = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and "teacher" not in n
                ]
            }
        ]
    else:
        parameter_groups = build_parameter_groups(
            model, _rule_value(cfg, "lr"), param_dicts
        )
        logger = get_root_logger()
        for index, group in enumerate(parameter_groups, start=1):
            message = ""
            for key in group:
                if key not in ("names", "params"):
                    message += f" {key}: {group[key]};"
            logger.info(
                f"Params Group {index} -{message} Params: {group['names']}."
            )

    optimizer_cfg["params"] = [
        {key: value for key, value in group.items() if key != "names"}
        for group in parameter_groups
    ]
    return OPTIMIZERS.build(cfg=optimizer_cfg)
