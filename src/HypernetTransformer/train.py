#!/usr/bin/env python3
"""
Train HyperNet GPT-2

Supports:
- --data_dir (WikiText-103 style folder) or --train_file (plain text)
- initialize from HF GPT-2 weights (--use_pretrained)
- save checkpoints and resume (--resume_from_checkpoint)
- optional orthogonal-projected optimizer (if orthogonal_projected_optimizer.py is present)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

# ---- Local imports ----
from hypernet_model import create_hypernet_gpt2

# Dataset helpers (your dataset.py)
# Dataset helpers (local dataset.py next to this script)
import importlib.util as _importlib_util

SCRIPT_DIR = Path(__file__).resolve().parent
_DATASET_IMPORT_ERROR = None
WikiTextDataset = None  # type: ignore
SimpleTextDataset = None  # type: ignore
HAS_DATASET = False

try:
    dataset_py = SCRIPT_DIR / "dataset.py"
    if dataset_py.exists():
        spec = _importlib_util.spec_from_file_location("hypernet_local_dataset", str(dataset_py))
        assert spec and spec.loader
        _mod = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(_mod)  # type: ignore
        WikiTextDataset = getattr(_mod, "WikiTextDataset", None)
        SimpleTextDataset = getattr(_mod, "SimpleTextDataset", None)
        HAS_DATASET = WikiTextDataset is not None or SimpleTextDataset is not None
    else:
        # Fallback to normal import if file is not present
        from dataset import WikiTextDataset as _WTD, SimpleTextDataset as _STD  # type: ignore
        WikiTextDataset, SimpleTextDataset = _WTD, _STD
        HAS_DATASET = True
except Exception as e:
    _DATASET_IMPORT_ERROR = e
    HAS_DATASET = False

# Orthogonal projection optimizer (optional)
OrthogonalProjectedAdamW = None
try:
    # Prefer local file next to train.py
    from orthogonal_projected_optimizer import OrthogonalProjectedAdamW  # type: ignore
except Exception:
    OrthogonalProjectedAdamW = None


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got: {v}")


class GateScheduler:
    def __init__(
        self,
        schedule_type: str,
        start_value: float,
        end_value: float,
        warmup_steps: int,
        total_steps: int,
    ):
        self.schedule_type = schedule_type
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.warmup_steps = int(max(warmup_steps, 0))
        self.total_steps = int(max(total_steps, 1))

    def get_gate_value(self, step: int) -> float:
        step = int(step)
        if self.schedule_type == "constant":
            return self.end_value

        if self.warmup_steps <= 0:
            t = min(max(step / self.total_steps, 0.0), 1.0)
        else:
            if step <= self.warmup_steps:
                t = step / self.warmup_steps
            else:
                t = min(max((step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1), 0.0), 1.0)

        if self.schedule_type == "linear":
            return self.start_value + t * (self.end_value - self.start_value)
        if self.schedule_type == "cosine":
            # cosine anneal from start -> end
            ct = 0.5 * (1 - math.cos(math.pi * t))
            return self.start_value + ct * (self.end_value - self.start_value)

        # fallback
        return self.end_value


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        lr: float,
        weight_decay: float,
        use_orthogonal_projection: bool,
        projection_mode: str,
    ):
        self.model = model
        self.device = device

        # Split params: (1) everything except centroids, (2) centroids-like parameters
        centroid_params = []
        main_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n.lower() for k in ["centroid", "centroids"]):
                centroid_params.append(p)
            else:
                main_params.append(p)

        OptimCls = torch.optim.AdamW
        if use_orthogonal_projection and OrthogonalProjectedAdamW is not None:
            OptimCls = OrthogonalProjectedAdamW

        if use_orthogonal_projection and OrthogonalProjectedAdamW is None:
            print("⚠️  Orthogonal projection optimizer not found. Using standard AdamW.")

        # Some implementations of OrthogonalProjectedAdamW may accept projection_mode
        def _make_optimizer(params):
            if OptimCls is OrthogonalProjectedAdamW:
                try:
                    return OptimCls(params, lr=lr, weight_decay=weight_decay, projection_mode=projection_mode)
                except TypeError:
                    return OptimCls(params, lr=lr, weight_decay=weight_decay)
            return OptimCls(params, lr=lr, weight_decay=weight_decay)

        self.main_optimizer = _make_optimizer(main_params)
        self.centroid_optimizer = _make_optimizer(centroid_params) if centroid_params else None

        self.main_scheduler = None
        self.centroid_scheduler = None

    def attach_schedulers(self, total_steps: int, warmup_steps: int):
        self.main_scheduler = get_linear_schedule_with_warmup(
            self.main_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        if self.centroid_optimizer is not None:
            self.centroid_scheduler = get_linear_schedule_with_warmup(
                self.centroid_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )

    def train_step(self, batch: Dict[str, torch.Tensor], gate_value: float) -> Dict[str, float]:
        self.model.train()

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            # force boolean mask for padding logic
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask != 0

        labels = input_ids.clone()
        if attention_mask is not None:
            # ignore pad tokens in loss
            labels = labels.masked_fill(~attention_mask, -100)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            gate_schedule=gate_value,
            return_dict=True,
        )
        loss = outputs.loss

        self.main_optimizer.zero_grad(set_to_none=True)
        if self.centroid_optimizer is not None:
            self.centroid_optimizer.zero_grad(set_to_none=True)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.main_optimizer.step()
        if self.centroid_optimizer is not None:
            self.centroid_optimizer.step()

        if self.main_scheduler is not None:
            self.main_scheduler.step()
        if self.centroid_scheduler is not None:
            self.centroid_scheduler.step()

        lm_loss = float(loss.detach().item())
        return {
            "lm_loss": lm_loss,
            "perplexity": float(math.exp(min(lm_loss, 20.0))),
            "gate_value": float(gate_value),
        }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    trainer: Trainer,
    epoch: int,
    global_step: int,
    extra: Optional[Dict[str, Any]] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "main_optimizer": trainer.main_optimizer.state_dict(),
        "main_scheduler": trainer.main_scheduler.state_dict() if trainer.main_scheduler else None,
        "centroid_optimizer": trainer.centroid_optimizer.state_dict() if trainer.centroid_optimizer else None,
        "centroid_scheduler": trainer.centroid_scheduler.state_dict() if trainer.centroid_scheduler else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "extra": extra or {},
    }
    torch.save(payload, str(path))


defdef = Tuple[torch.nn.Module, Optional[Any]]


def load_checkpoint(path: Path, model: torch.nn.Module, trainer: Trainer) -> Tuple[int, int]:
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    trainer.main_optimizer.load_state_dict(ckpt["main_optimizer"])
    if trainer.main_scheduler is not None and ckpt.get("main_scheduler") is not None:
        trainer.main_scheduler.load_state_dict(ckpt["main_scheduler"])

    if trainer.centroid_optimizer is not None and ckpt.get("centroid_optimizer") is not None:
        trainer.centroid_optimizer.load_state_dict(ckpt["centroid_optimizer"])
    if trainer.centroid_scheduler is not None and ckpt.get("centroid_scheduler") is not None:
        trainer.centroid_scheduler.load_state_dict(ckpt["centroid_scheduler"])

    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))


def build_model(args, tokenizer: GPT2Tokenizer, device: torch.device) -> torch.nn.Module:
    result = create_hypernet_gpt2(
        base_model=args.base_model,
        use_pretrained=args.use_pretrained,
        num_centroids=args.num_centroids,
        memory_window=args.memory_window,
        reasoning_json_path=args.reasoning_json_path,
        use_rl=False,
        rl_method=args.rl_method,
        num_layers=getattr(args, "num_layers", 12),
        hypernet_start_layer=getattr(args, "hypernet_start_layer", 12),
    )

    # Robust: accept either model OR (model, config)
    if isinstance(result, tuple) and len(result) >= 1:
        model = result[0]
    else:
        model = result

    if hasattr(model, "set_tokenizer"):
        model.set_tokenizer(tokenizer)

    model = model.to(device)
    return model


def build_dataset(args, tokenizer: GPT2Tokenizer):
    if args.data_dir:
        if not HAS_DATASET or WikiTextDataset is None:
            raise RuntimeError(f"Could not load WikiTextDataset from local dataset.py. Import error: {_DATASET_IMPORT_ERROR}")
        return WikiTextDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            split="train",
            max_length=args.max_length,
        )

    if args.train_file:
        if not HAS_DATASET or SimpleTextDataset is None:
            raise RuntimeError("dataset.py did not import SimpleTextDataset; please check dataset.py.")
        return SimpleTextDataset(
            args.train_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

    raise ValueError("Must provide either --data_dir or --train_file")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_dir", type=str, default=None, help="Dataset directory (e.g., WikiText-103).")
    p.add_argument("--train_file", type=str, default=None, help="Plaintext file path for SimpleTextDataset.")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)

    # Model
    p.add_argument("--base_model", type=str, default="gpt2")
    p.add_argument("--use_pretrained", action="store_true", help="Initialize from HF pretrained GPT-2 weights.")
    p.add_argument("--num_centroids", type=int, default=128)
    p.add_argument("--memory_window", type=int, default=10)
    p.add_argument("--reasoning_json_path", type=str, default="reasoning_logic.json")
    p.add_argument("--rl_method", type=str, default="actor_critic")

    # Train
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=200)

    # Gate schedule
    p.add_argument("--gate_schedule", type=str, default="linear", choices=["linear", "cosine", "constant"])
    p.add_argument("--gate_start", type=float, default=0.0)
    p.add_argument("--gate_end", type=float, default=1.0)

    # Checkpointing
    p.add_argument("--output_dir", type=str, default="./out")
    p.add_argument("--save_every", type=int, default=200, help="Save checkpoint every N steps.")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Optimizer options
    p.add_argument("--use_orthogonal_projection", action="store_true")
    p.add_argument("--projection_mode", type=str, default="default")

    return p.parse_args()


def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    print("📦 Building model...")
    model = build_model(args, tokenizer, device)

    # Dataset
    print("📚 Loading dataset...")
    train_dataset = build_dataset(args, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"✅ Dataset loaded: {len(train_dataset)} samples")

    total_steps = len(train_loader) * args.num_epochs
    gate_scheduler = GateScheduler(
        schedule_type=args.gate_schedule,
        start_value=args.gate_start,
        end_value=args.gate_end,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_orthogonal_projection=args.use_orthogonal_projection,
        projection_mode=args.projection_mode,
    )
    trainer.attach_schedulers(total_steps=total_steps, warmup_steps=args.warmup_steps)

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        ckpt_path = Path(args.resume_from_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume_from_checkpoint not found: {ckpt_path}")
        print(f"🔄 Resuming from checkpoint: {ckpt_path}")
        start_epoch, global_step = load_checkpoint(ckpt_path, model, trainer)
        print(f"✅ Resumed at epoch={start_epoch}, global_step={global_step}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n🚀 Training")
    print(f"   epochs: {args.num_epochs}")
    print(f"   steps/epoch: {len(train_loader)}")
    print(f"   total steps: {total_steps}")
    print(f"   save_every: {args.save_every}")
    print(f"   output_dir: {output_dir}\n")

    for epoch in range(start_epoch, args.num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            gate_value = gate_scheduler.get_gate_value(global_step)

            metrics = trainer.train_step(batch, gate_value=gate_value)
            running_loss += metrics["lm_loss"]

            if global_step % 20 == 0:
                avg = running_loss / 20.0
                running_loss = 0.0
                print(
                    f"epoch {epoch+1}/{args.num_epochs} | step {global_step}/{total_steps} "
                    f"| loss {avg:.4f} | ppl {math.exp(min(avg, 20.0)):.2f} | gate {gate_value:.3f}"
                )

            if args.save_every > 0 and (global_step % args.save_every == 0):
                ckpt = output_dir / f"checkpoint-{global_step}.pt"
                save_checkpoint(ckpt, model, trainer, epoch=epoch, global_step=global_step, extra={"args": vars(args)})
                print(f"💾 Saved {ckpt}")

        # epoch-end checkpoint
        ckpt = output_dir / f"checkpoint-epoch{epoch+1}-step{global_step}.pt"
        save_checkpoint(ckpt, model, trainer, epoch=epoch+1, global_step=global_step, extra={"args": vars(args)})
        print(f"💾 Saved {ckpt}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
