from abc import ABC, abstractmethod
import argparse
import os
from typing import Any, Dict, Iterable, List, Tuple
from vq_voice_swap.loss_tracker import LossTracker

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .dataset import create_data_loader
from .diffusion import Diffusion, make_schedule
from .diffusion_model import DiffusionModel
from .ema import ModelEMA
from .logger import Logger
from .loss_tracker import LossTracker
from .models import Savable, Classifier
from .util import count_params, repeat_dataset
from .vq_vae import VQVAE


class TrainLoop(ABC):
    """
    An abstract training loop with methods to override for controlling
    different pieces of training.
    """

    def __init__(self, args=None):
        if args is None:
            args = self.arg_parser().parse_args()
        self.args = args

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        self.data_loader, self.num_labels = self.create_data_loader()
        self.model, self.resume = self.create_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.ema = self.create_ema()
        self.opt = self.create_opt()
        self.logger, self.tracker = self.create_logger_tracker()

        self.total_steps = self.logger.start_step
        self.loop_steps = 0

    def loop(self):
        for i, data_batch in enumerate(repeat_dataset(self.data_loader)):
            self.total_steps = i + self.logger.start_step
            self.loop_steps = i
            self.step(data_batch)

    def step(self, data_batch: Dict[str, torch.Tensor]):
        self.opt.zero_grad()

        all_losses = []
        all_ts = []
        all_loss = 0.0
        all_extra = dict()

        for microbatch, weight in self.split_microbatches(data_batch):
            losses, ts, extra_losses = self.compute_losses(microbatch)

            # Re-weighted losses for microbatch averaging
            extra_losses = {k: v * weight for k, v in extra_losses.items()}
            loss = losses.mean() * weight
            for extra in extra_losses.values():
                loss = loss + extra

            self.loss_backward(loss)

            # Needed to re-aggregate the microbatch losses for
            # normal logging.
            all_losses.append(losses.detach())
            all_ts.append(ts)
            all_loss = all_loss + loss.detach()
            all_extra = {
                k: v.detach() + all_extra.get(k, 0.0) for k, v in extra_losses.items()
            }

        self.step_optimizer()
        self.log_losses(
            all_loss, torch.cat(all_losses, dim=0), torch.cat(all_ts, dim=0), all_extra
        )

        if (self.total_steps + 1) % self.args.save_interval == 0:
            self.model.save(self.checkpoint_path())
            self.ema.model.save(self.ema_path())

    def split_microbatches(
        self, data_batch: Dict[str, torch.Tensor]
    ) -> List[Tuple[Dict[str, torch.Tensor], float]]:
        key = next(iter(data_batch.keys()))
        batch_size = len(data_batch[key])
        if not self.args.microbatch or self.args.microbatch > batch_size:
            return [(data_batch, 1.0)]
        res = []
        for i in range(0, batch_size, self.args.microbatch):
            sub_batch = {
                k: v[i : i + self.args.microbatch] for k, v in data_batch.items()
            }
            res.append((sub_batch, len(sub_batch[key]) / batch_size))
        return res

    def loss_backward(self, loss: torch.Tensor):
        loss.backward()

    def step_optimizer(self):
        self.opt.step()
        self.ema.update()

    @abstractmethod
    def compute_losses(
        self, data_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss per batch element, and also return the diffusion timestep
        for each loss. Also return a (possibly empty) dict of other losses.

        :return: a tuple (losses, ts, other).
        """

    def log_losses(
        self,
        loss: torch.Tensor,
        losses: torch.Tensor,
        ts: torch.Tensor,
        extra_losses: Dict[str, torch.Tensor],
    ):
        self.tracker.add(ts, losses)
        other = {k: v.item() for k, v in extra_losses.items()}
        other.update(self.tracker.log_dict())
        self.logger.log(self.loop_steps + 1, loss=loss.item(), **other)

    def create_data_loader(self) -> Tuple[Iterable, int]:
        return create_data_loader(
            directory=self.args.data_dir,
            batch_size=self.args.batch_size,
            encoding=self.args.encoding,
        )

    def create_model(self) -> Tuple[Savable, bool]:
        if os.path.exists(self.checkpoint_path()):
            print("loading from checkpoint...")
            model = self.model_class().load(self.checkpoint_path())
            resume = True
        else:
            print("creating new model")
            model = self.create_new_model()
            resume = False

            if self.args.pretrained_path:
                print(f"loading from pretrained model: {self.args.pretrained_path} ...")
                num_params = self.load_from_pretrained(model)
                print(f"loaded {num_params} pre-trained parameters...")
        print(f"total parameters: {count_params(model)}")
        return model, resume

    def create_ema(self) -> ModelEMA:
        ema = ModelEMA(self.model, rates={"": self.args.ema_rate})
        if os.path.exists(self.ema_path()):
            print("loading EMA from checkpoint...")
            ema.model = self.model_class().load(self.ema_path()).to(self.device)
        return ema

    def create_opt(self) -> torch.optim.Optimizer:
        return AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def create_logger_tracker(self) -> Tuple[Logger, LossTracker]:
        return Logger(self.log_path(), resume=self.resume), LossTracker()

    def checkpoint_path(self):
        return os.path.join(self.args.output_dir, "model.pt")

    def ema_path(self):
        return os.path.join(self.args.output_dir, "model_ema.pt")

    def log_path(self):
        return os.path.join(self.args.output_dir, "train_log.txt")

    @abstractmethod
    def model_class(self) -> Any:
        """
        Get the Savable class used to construct models.
        """

    @abstractmethod
    def create_new_model(self) -> Savable:
        """
        Create a new instance of the model.
        """

    def load_from_pretrained(self, model: Savable) -> int:
        pt = self.model_class().load(self.args.pretrained_path)
        return model.load_from_pretrained(pt)

    @classmethod
    def arg_parser(cls) -> argparse.ArgumentParser:
        """
        Get an argument parser for the training command.
        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument("--ema-rate", default=0.9999, type=float)
        parser.add_argument("--weight-decay", default=0.0, type=float)
        parser.add_argument("--batch-size", default=8, type=int)
        parser.add_argument("--microbatch", default=None, type=int)
        parser.add_argument("--output-dir", default=cls.default_output_dir(), type=str)
        parser.add_argument("--pretrained-path", default=None, type=str)
        parser.add_argument("--save-interval", default=1000, type=int)
        parser.add_argument("--grad-checkpoint", action="store_true")
        parser.add_argument("--encoding", default="linear", type=str)
        parser.add_argument("data_dir", type=str)
        return parser

    @classmethod
    @abstractmethod
    def default_output_dir(cls) -> str:
        """
        Get the default directory name for training output.
        """


class DiffusionTrainLoop(TrainLoop):
    def compute_losses(
        self, data_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        audio_seq = data_batch["samples"][:, None].to(self.device)
        if self.args.class_cond:
            extra_kwargs = dict(labels=data_batch["label"].to(self.device))
        else:
            extra_kwargs = dict()
        ts = torch.rand(len(audio_seq), device=self.device)
        losses = self.model.diffusion.ddpm_losses(
            audio_seq,
            self.model.predictor.condition(
                use_checkpoint=self.args.grad_checkpoint, **extra_kwargs
            ),
            ts=ts,
        )
        return losses, ts, dict()

    def model_class(self) -> Any:
        return DiffusionModel

    def create_new_model(self) -> Savable:
        return self.model_class()(
            pred_name=self.args.predictor,
            base_channels=self.args.base_channels,
            schedule_name=self.args.schedule,
            dropout=self.args.dropout,
            num_labels=self.num_labels if self.args.class_cond else None,
        )

    @classmethod
    def arg_parser(cls) -> argparse.ArgumentParser:
        parser = super().arg_parser()
        parser.add_argument("--predictor", default="unet", type=str)
        parser.add_argument("--base-channels", default=32, type=int)
        parser.add_argument("--dropout", default=0.0, type=float)
        parser.add_argument("--schedule", default="exp", type=str)
        parser.add_argument("--class-cond", action="store_true")
        return parser

    @classmethod
    def default_output_dir(cls) -> str:
        return "ckpt_diffusion"


class VQVAETrainLoop(DiffusionTrainLoop):
    def compute_losses(
        self, data_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        audio_seq = data_batch["samples"][:, None].to(self.device)
        if self.args.class_cond:
            extra_kwargs = dict(labels=data_batch["label"].to(self.device))
        else:
            extra_kwargs = dict()
        losses = self.model.losses(
            audio_seq, **extra_kwargs, use_checkpoint=self.args.grad_checkpoint
        )
        return losses["mses"], losses["ts"], dict(vq_loss=losses["vq_loss"])

    def model_class(self) -> Any:
        return VQVAE

    def step_optimizer(self):
        super().step_optimizer()
        self.model.vq.revive_dead_entries()

    @classmethod
    def default_output_dir(cls) -> str:
        return "ckpt_vqvae"


class ClassifierTrainLoop(TrainLoop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diffusion = Diffusion(make_schedule(self.args.schedule))

    def compute_losses(
        self, data_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        audio_seq = data_batch["samples"][:, None].to(self.device)
        labels = data_batch["label"].to(self.device)
        ts = self.sample_timesteps(len(audio_seq))

        samples = self.diffusion.sample_q(audio_seq, ts)
        logits = self.model(samples, ts, use_checkpoint=self.args.grad_checkpoint)
        nlls = -F.log_softmax(logits, dim=-1)[range(len(labels)), labels]
        return nlls, ts, dict()

    def sample_timesteps(self, n: int) -> torch.Tensor:
        ts = torch.rand(n, device=self.device)
        if self.total_steps < self.args.curriculum_steps:
            frac = self.total_steps / self.args.curriculum_steps
            power = self.args.curriculum_start * (1 - frac) + frac
            ts = ts ** power
        return ts

    def model_class(self) -> Any:
        return Classifier

    def create_new_model(self) -> Savable:
        return self.model_class()(
            num_labels=self.num_labels, base_channels=self.args.base_channels
        )

    def load_from_pretrained(self, model: Savable) -> int:
        dm = DiffusionModel.load(self.args.pretrained_path)
        return model.load_from_predictor(dm.predictor)

    @classmethod
    def arg_parser(cls) -> argparse.ArgumentParser:
        parser = super().arg_parser()
        parser.add_argument("--base-channels", default=32, type=int)
        parser.add_argument("--schedule", default="exp", type=str)
        parser.add_argument("--curriculum-start", default=30.0, type=float)
        parser.add_argument("--curriculum-steps", default=0, type=int)
        return parser

    @classmethod
    def default_output_dir(cls) -> str:
        return "ckpt_classifier"
