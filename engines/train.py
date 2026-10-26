import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial
from pathlib import Path

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import utils.comm as comm
from datasets import build_dataset, point_collate_fn, collate_fn
from models import build_model
from utils.logger import get_root_logger
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.events import EventStorage, ExceptionWriter
from utils.registry import Registry
from utils.dataloader import build_loader_runtime_options
from utils.debug_indices import assert_finite, debug_indices_enabled
from utils.ema_distill import (
    confidence_gated_kl,
    cosine_ema_distill_weight,
    ema_teacher_due,
)

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)


            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:

            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):

                self.before_epoch()

                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:

                    self.before_step()

                    self.run_step()

                    self.after_step()

                self.after_epoch()

            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):

        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if self.writer is not None and comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        sharing_strategy = getattr(cfg, "multiprocessing_sharing_strategy", None)
        if sharing_strategy is not None:
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.best_ema_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.debug_indices = debug_indices_enabled()
        if self.debug_indices:
            torch.autograd.set_detect_anomaly(True)
            self.logger.warning(
                "[LITEPT_DEBUG_INDICES] enabled; CUDA_LAUNCH_BLOCKING=%s",
                os.environ.get("CUDA_LAUNCH_BLOCKING", "<unset>"),
            )
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():

            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):

                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()

                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:

                    self.before_step()

                    self.run_step()

                    self.after_step()

                self.after_epoch()

            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:

            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        global_step = self.epoch * len(self.train_loader) + self.comm_info["iter"]
        distill_weight = self._ema_distill_weight(global_step)
        distill_cfg = getattr(self.cfg, "ema_distill", None)
        teacher_logits = None
        if distill_cfg is not None and ema_teacher_due(
            global_step,
            int(distill_cfg.get("interval", 4)),
            distill_weight,
        ):
            teacher_logits = self._ema_teacher_logits(input_dict, auto_cast)

        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            if teacher_logits is None:
                output_dict = self.model(input_dict)
            else:
                output_dict = self.model(input_dict, return_logits=True)
            loss = output_dict["loss"]
            if teacher_logits is not None:
                student_logits = output_dict.pop("seg_logits")
                geometry_reliability = output_dict.pop(
                    "geometry_reliability", None
                )
                consistency_loss, coverage = confidence_gated_kl(
                    student_logits,
                    teacher_logits,
                    input_dict["segment"],
                    confidence_threshold=float(
                        distill_cfg.get("confidence_threshold", 0.8)
                    ),
                    ignore_index=int(distill_cfg.get("ignore_index", -1)),
                    geometry_reliability=geometry_reliability,
                    reliability_floor=float(
                        distill_cfg.get("reliability_floor", 0.0)
                    ),
                )
                loss = loss + distill_weight * consistency_loss
                output_dict["loss"] = loss
                output_dict["ema_consistency_loss"] = consistency_loss.detach()
                output_dict["ema_distill_weight"] = loss.new_tensor(distill_weight)
                output_dict["ema_teacher_coverage"] = coverage.detach()
        if self.debug_indices and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            if self.debug_indices and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.scaler.unscale_(self.optimizer)
            self._debug_assert_grads_finite("after_unscale")
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)
            self._debug_assert_params_finite("after_optimizer_step")


            old_scale = self.scaler.get_scale()
            self.scaler.update()
            step_succeeded = old_scale <= self.scaler.get_scale()
            self.comm_info["optimizer_step_succeeded"] = step_succeeded
            if step_succeeded:
                self.scheduler.step()
        else:
            loss.backward()
            if self.debug_indices and torch.cuda.is_available():
                torch.cuda.synchronize()
            self._debug_assert_grads_finite("after_backward")
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self._debug_assert_params_finite("after_optimizer_step")
            self.comm_info["optimizer_step_succeeded"] = True
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def _ema_distill_weight(self, global_step):
        cfg = getattr(self.cfg, "ema_distill", None)
        if (
            cfg is None
            or not cfg.get("enabled", False)
            or not hasattr(self, "model_ema")
        ):
            return 0.0
        total_steps = max(int(self.cfg.scheduler.total_steps) - 1, 1)
        return cosine_ema_distill_weight(
            progress=float(global_step) / total_steps,
            max_weight=float(cfg.get("max_weight", 0.05)),
            final_weight=float(cfg.get("final_weight", 0.02)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
            ramp_end_ratio=float(cfg.get("ramp_end_ratio", 0.2)),
            decay_start_ratio=float(cfg.get("decay_start_ratio", 0.8)),
        )

    def _ema_teacher_logits(self, input_dict, auto_cast):
        if not hasattr(self, "model_ema"):
            raise RuntimeError("EMA distillation requires ModelEMAHook")

        teacher_input = dict(input_dict)
        teacher_input.pop("segment", None)
        if not self.model.training:
            raise RuntimeError("EMA teacher must reuse the training-mode sparse path")
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            with torch.no_grad():
                with self.model_ema.average_parameters(self.model):
                    with auto_cast(
                        enabled=self.cfg.enable_amp,
                        dtype=AMP_DTYPE[self.cfg.amp_dtype],
                    ):
                        output_dict = self.model(teacher_input, logits_only=True)
        return output_dict["seg_logits"].detach()

    def _debug_assert_grads_finite(self, where):
        if not self.debug_indices:
            return
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                assert_finite(f"{where}.grad.{name}", parameter.grad)

    def _debug_assert_params_finite(self, where):
        if not self.debug_indices:
            return
        for name, parameter in self.model.named_parameters():
            assert_finite(f"{where}.param.{name}", parameter)

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            import wandb

            tag, name = Path(self.cfg.save_path).parts[-2:]
            if self.cfg.resume:
                wandb.init(
                    project=self.cfg.wandb_project,
                    name=f"{tag}/{name}",
                    tags=[tag],

                    dir=self.cfg.save_path,
                    settings=wandb.Settings(api_key=self.cfg.wandb_key),
                    config=self.cfg,
                )
            else:
                wandb.init(
                    project=self.cfg.wandb_project,
                    name=f"{tag}/{name}",
                    tags=[tag],
                    dir=self.cfg.save_path,
                    settings=wandb.Settings(api_key=self.cfg.wandb_key),
                    config=self.cfg,
                )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            **build_loader_runtime_options(self.cfg, self.cfg.num_worker_per_gpu),
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                sampler=val_sampler,
                collate_fn=collate_fn,
                **build_loader_runtime_options(self.cfg, self.cfg.num_worker_per_gpu),
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:

            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
            runtime_cfg=self.cfg,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
