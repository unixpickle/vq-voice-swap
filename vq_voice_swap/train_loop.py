from abc import ABC, abstractmethod
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Set, Tuple
from vq_voice_swap.loss_tracker import LossTracker

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from .dataset import create_data_loader
from .diffusion import Diffusion, make_schedule
from .diffusion_model import DiffusionModel
from .ema import ModelEMA
from .logger import Logger
from .loss_tracker import LossTracker
from .models import Savable, Classifier, EncoderPredictor
from .util import count_params, repeat_dataset
from .vq import StandardVQLoss, ReviveVQLoss
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        self.data_loader, self.num_labels = self.create_data_loader()
        self.model, self.resume = self.create_model()
        self.model.to(self.device)

        self.emas = self.create_emas()
        self.opt = self.create_opt()
        self.logger, self.tracker = self.create_logger_tracker()

        self.total_steps = self.logger.start_step
        self.loop_steps = 0

        self.freeze_parameters(self.frozen_parameters())
        self.write_run_info()

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
            self.save()

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
        for ema in self.emas.values():
            ema.update()

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

    def save(self):
        self.model.save(self.checkpoint_path())
        for rate, ema in self.emas.items():
            ema.model.save(self.ema_path(rate))
        torch.save(self.opt.state_dict(), self.opt_path())
        self.logger.mark_save()

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

    def create_emas(self) -> Dict[float, ModelEMA]:
        res = {}
        for rate_str in self.args.ema_rate.split(","):
            rate = float(rate_str)
            assert rate not in res, "cannot have duplicate EMA rate"
            ema = ModelEMA(self.model, rates={"": rate})
            path = self.ema_path(rate)
            if os.path.exists(path):
                print(f"loading EMA {rate} from checkpoint...")
                ema.model = self.model_class().load(path).to(self.device)
            res[rate] = ema
        return res

    def create_opt(self) -> torch.optim.Optimizer:
        opt = AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        if os.path.exists(self.opt_path()):
            print("loading optimizer from checkpoint...")
            opt.load_state_dict(torch.load(self.opt_path(), map_location="cpu"))
        return opt

    def frozen_parameters(self) -> Set[nn.Parameter]:
        return set()

    def freeze_parameters(self, params: Set[nn.Parameter]):
        param_to_idx = {param: idx for idx, param in enumerate(self.model.parameters())}
        count = 0
        sd = self.opt.state_dict().copy()
        for p in params:
            self.freeze_parameter(param_to_idx[p], p, sd)
            count += p.numel()
        if count:
            self.opt.load_state_dict(sd)
            print(f"frozen parameters: {count}")

    def freeze_parameter(
        self, idx: int, param: nn.Parameter, opt_state: Dict[str, Any]
    ):
        param.requires_grad_(False)
        if idx in opt_state["state"]:
            # A step has been taken, and the parameter might have some
            # momentum built up that we need to cancel out.
            assert opt_state["state"][idx]["exp_avg"].shape == param.shape
            opt_state["state"] = opt_state["state"].copy()
            opt_state["state"][idx] = opt_state["state"][idx].copy()
            opt_state["state"][idx]["exp_avg"].zero_()
            opt_state["state"][idx]["exp_avg_sq"].zero_()

    def create_logger_tracker(self) -> Tuple[Logger, LossTracker]:
        return Logger(self.log_path(), resume=self.resume), LossTracker()

    def checkpoint_path(self):
        return os.path.join(self.args.output_dir, "model.pt")

    def ema_path(self, rate):
        return os.path.join(self.args.output_dir, f"model_ema_{rate}.pt")

    def opt_path(self):
        return os.path.join(self.args.output_dir, "opt.pt")

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

    def write_run_info(self):
        filename = f"run_info_{int(time.time())}.json"
        with open(os.path.join(self.args.output_dir, filename), "w+") as f:
            json.dump(self.run_info(), f, indent=4)

    def run_info(self) -> Dict:
        return dict(
            args=self.args.__dict__,
            command=sys.argv[0],
            start_steps=self.total_steps,
        )

    @classmethod
    def arg_parser(cls) -> argparse.ArgumentParser:
        """
        Get an argument parser for the training command.
        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument("--ema-rate", default="0.9999", type=str)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.args.revival_coeff:
            self.vq_loss = ReviveVQLoss(
                revival=self.args.revival_coeff, commitment=self.args.commitment_coeff
            )
        else:
            self.vq_loss = StandardVQLoss(commitment=self.args.commitment_coeff)

    def compute_losses(
        self, data_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        audio_seq = data_batch["samples"][:, None].to(self.device)
        if self.args.class_cond:
            extra_kwargs = dict(labels=data_batch["label"].to(self.device))
        else:
            extra_kwargs = dict()
        losses = self.model.losses(
            self.vq_loss,
            audio_seq,
            jitter=self.args.jitter,
            **extra_kwargs,
            use_checkpoint=self.args.grad_checkpoint,
        )
        return losses["mses"], losses["ts"], dict(vq_loss=losses["vq_loss"])

    def model_class(self) -> Any:
        return VQVAE

    def create_model(self) -> Tuple[Savable, bool]:
        model, resume = super().create_model()
        model.vq.dead_rate = self.args.dead_rate
        return model, resume

    def create_new_model(self) -> Savable:
        return self.model_class()(
            pred_name=self.args.predictor,
            base_channels=self.args.base_channels,
            enc_name=self.args.encoder,
            cond_mult=self.args.cond_mult,
            dictionary_size=self.args.dictionary_size,
            schedule_name=self.args.schedule,
            dropout=self.args.dropout,
            num_labels=self.num_labels if self.args.class_cond else None,
        )

    def frozen_parameters(self) -> Set[nn.Parameter]:
        res = set()
        if self.args.freeze_encoder:
            res.update(self.model.encoder.parameters())
        if self.args.freeze_vq:
            res.update(self.model.vq.parameters())
        return res

    @classmethod
    def arg_parser(cls) -> argparse.ArgumentParser:
        parser = super().arg_parser()
        parser.add_argument("--encoder", default="unet", type=str)
        parser.add_argument("--cond-mult", default=16, type=int)
        parser.add_argument("--dictionary-size", default=512, type=int)
        parser.add_argument("--freeze-encoder", action="store_true")
        parser.add_argument("--freeze-vq", action="store_true")
        parser.add_argument("--commitment-coeff", default=0.25, type=float)
        parser.add_argument("--revival-coeff", default=0.0, type=float)
        parser.add_argument("--dead-rate", default=100, type=int)
        parser.add_argument("--jitter", default=0.0, type=float)
        return parser

    def load_from_pretrained(self, model: Savable) -> int:
        pt, err = None, None
        for cls in [self.model_class(), DiffusionModel]:
            try:
                pt = cls.load(self.args.pretrained_path)
            except RuntimeError as exc:
                err = exc
        if pt is None:
            raise err
        return model.load_from_pretrained(pt)

    def step_optimizer(self):
        super().step_optimizer()
        if self.should_revive():
            self.model.vq.revive_dead_entries()

    def should_revive(self) -> bool:
        return not self.args.revival_coeff and not self.args.freeze_vq

    @classmethod
    def default_output_dir(cls) -> str:
        return "ckpt_vqvae"


class VQVAEAddClassesTrainLoop(VQVAETrainLoop):
    def __init__(self, **kwargs):
        # These are set during model load.
        self.pretrained_kwargs = None
        self.pretrained_num_labels = None

        super().__init__(**kwargs)
        assert self.args.class_cond

    def compute_losses(
        self, data_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        data_batch["label"] = data_batch["label"] + self.pretrained_num_labels
        return super().compute_losses(data_batch)

    def create_model(self) -> Tuple[Savable, bool]:
        assert self.args.pretrained_path, "must load from a pre-trained VQVAE"
        assert self.args.class_cond, "must create a class-conditional model"
        pretrained = VQVAE.load(self.args.pretrained_path)
        self.pretrained_num_labels = pretrained.num_labels
        self.pretrained_kwargs = pretrained.save_kwargs()

        return super().create_model()

    def create_new_model(self) -> Savable:
        kwargs = self.pretrained_kwargs.copy()
        kwargs["num_labels"] = self.num_labels + self.pretrained_num_labels
        return self.model_class()(**kwargs)

    def load_from_pretrained(self, model: Savable) -> int:
        base_model = VQVAE.load(self.args.pretrained_path)
        base_model.add_labels(self.num_labels)
        return model.load_from_pretrained(base_model)

    def frozen_parameters(self) -> Set[nn.Parameter]:
        label_params = set(self.model.predictor.label_parameters())
        x = set(x for x in self.model.parameters() if x not in label_params)
        return x

    def should_revive(self) -> bool:
        # Don't mess with the VQ codebook, since we might not be
        # using all of it for the new classes, but still want to
        # preserve functionality on the old classes.
        return False

    @classmethod
    def default_output_dir(cls) -> str:
        return "ckpt_vqvae_added"


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


class EncoderPredictorTrainLoop(TrainLoop):
    def __init__(self, **kwargs):
        self.vq_vae = None
        super().__init__(**kwargs)

    def compute_losses(
        self, data_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        audio_seq = data_batch["samples"][:, None].to(self.device)
        ts = self.sample_timesteps(len(audio_seq))
        with torch.no_grad():
            targets = self.vq_vae.encode(audio_seq)
        samples = self.vq_vae.diffusion.sample_q(audio_seq, ts)
        losses = self.model.losses(
            samples, ts, targets, use_checkpoint=self.args.grad_checkpoint
        )
        return losses, ts, dict()

    def sample_timesteps(self, n: int) -> torch.Tensor:
        ts = torch.rand(n, device=self.device)
        if self.total_steps < self.args.curriculum_steps:
            frac = self.total_steps / self.args.curriculum_steps
            power = self.args.curriculum_start * (1 - frac) + frac
            ts = ts ** power
        return ts

    def model_class(self) -> Any:
        return EncoderPredictor

    def create_model(self) -> Tuple[Savable, bool]:
        self.vq_vae = VQVAE.load(self.args.vq_vae_path).to(self.device)
        return super().create_model()

    def create_new_model(self) -> Savable:
        return self.model_class()(
            base_channels=self.args.base_channels,
            downsample_rate=self.vq_vae.encoder.downsample_rate,
            num_latents=self.vq_vae.dictionary_size,
        )

    @classmethod
    def arg_parser(cls) -> argparse.ArgumentParser:
        parser = super().arg_parser()
        parser.add_argument("--vq-vae-path", type=str, required=True)
        parser.add_argument("--base-channels", type=int, default=32)
        parser.add_argument("--curriculum-start", default=30.0, type=float)
        parser.add_argument("--curriculum-steps", default=0, type=int)
        return parser

    @classmethod
    def default_output_dir(cls) -> str:
        return "ckpt_enc_pred"
