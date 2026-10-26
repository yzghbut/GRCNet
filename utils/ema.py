# EMA weights track the training model without gradient updates.
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import math

import torch


class ModelEMA:
    def __init__(
        self,
        model,
        decay=0.9999,
        warmup=False,
        late_decay=None,
        late_decay_start_ratio=1.0,
        late_decay_end_ratio=1.0,
        total_updates=None,
    ):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        if late_decay is not None and not 0.0 <= late_decay < 1.0:
            raise ValueError("EMA late_decay must be in [0, 1)")
        if not 0.0 <= float(late_decay_start_ratio) <= 1.0:
            raise ValueError("EMA late_decay_start_ratio must be in [0, 1]")
        if not 0.0 <= float(late_decay_end_ratio) <= 1.0:
            raise ValueError("EMA late_decay_end_ratio must be in [0, 1]")
        if float(late_decay_start_ratio) > float(late_decay_end_ratio):
            raise ValueError("EMA late decay ratios must be ordered")
        if total_updates is not None and int(total_updates) <= 0:
            raise ValueError("EMA total_updates must be positive")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.late_decay = None if late_decay is None else float(late_decay)
        self.late_decay_start_ratio = float(late_decay_start_ratio)
        self.late_decay_end_ratio = float(late_decay_end_ratio)
        self.total_updates = None if total_updates is None else int(total_updates)
        self.num_updates = 0
        self.state = self._clone_state(model.state_dict())

    @property
    def current_decay(self):
        decay = self.decay
        if self.warmup:
            decay = min(
            self.decay,
            (1.0 + self.num_updates) / (10.0 + self.num_updates),
            )
        if (
            self.late_decay is None
            or self.total_updates is None
            or self.late_decay_start_ratio >= 1.0
        ):
            return decay
        progress = min(
            max(float(self.num_updates) / max(self.total_updates - 1, 1), 0.0),
            1.0,
        )
        start = self.late_decay_start_ratio
        end = self.late_decay_end_ratio
        if progress <= start:
            return decay
        if end <= start or progress >= end:
            return self.late_decay
        phase = (progress - start) / (end - start)

        blend = 0.5 * (1.0 - math.cos(math.pi * phase))
        return decay + (self.late_decay - decay) * blend

    @staticmethod
    def _clone_state(state):
        cloned = OrderedDict(
            (key, value.detach().clone()) for key, value in state.items()
        )
        if hasattr(state, "_metadata"):
            cloned._metadata = deepcopy(state._metadata)
        return cloned

    @staticmethod
    @torch.no_grad()
    def _copy_state(destination, source):
        if tuple(destination) != tuple(source):
            raise ValueError("state keys do not match")
        for key, value in source.items():
            if value.shape != destination[key].shape:
                raise ValueError(f"state tensor shape mismatch for {key}")
        for key, value in source.items():
            destination_value = destination[key]
            destination_value.copy_(
                value.detach().to(
                    device=destination_value.device,
                    dtype=destination_value.dtype,
                )
            )

    def _validate(self, state):
        if tuple(state) != tuple(self.state):
            raise ValueError("EMA state keys do not match model state keys")
        for key, value in state.items():
            if value.shape != self.state[key].shape:
                raise ValueError(f"EMA tensor shape mismatch for {key}")

    @torch.no_grad()
    def update(self, model):
        current = model.state_dict()
        self._validate(current)
        self.num_updates += 1
        decay = self.current_decay
        for key, value in current.items():
            ema_value = self.state[key]
            value = value.detach().to(
                device=ema_value.device,
                dtype=ema_value.dtype,
            )
            if torch.is_floating_point(ema_value) or torch.is_complex(ema_value):
                ema_value.mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                ema_value.copy_(value)

    def state_dict(self):
        return self._clone_state(self.state)

    @torch.no_grad()
    def load_state_dict(self, state, num_updates=None):
        self._validate(state)
        self._copy_state(self.state, state)
        if num_updates is not None:
            if isinstance(num_updates, bool) or not isinstance(num_updates, int):
                raise TypeError("EMA num_updates must be a non-negative integer")
            if num_updates < 0:
                raise ValueError("EMA num_updates must be non-negative")
            self.num_updates = num_updates

    @contextmanager
    def average_parameters(self, model):
        live_state = model.state_dict()
        self._validate(live_state)
        backup = self._clone_state(live_state)
        try:
            self._copy_state(live_state, self.state)
            yield
        finally:
            self._copy_state(live_state, backup)
