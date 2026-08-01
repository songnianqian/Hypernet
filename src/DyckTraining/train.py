#!/usr/bin/env python3
"""
Train HyperNet GPT-2 - Improved Version

Improvements:
1. Checkpoint save/load as separate functions
2. Save checkpoint before training to test preload correctness  
3. Add evaluation with PPL and accuracy calculation
4. Add prompt test during evaluation with generation sample
5. Training interrupt handling with graceful checkpoint save
6. Centroid win monitoring (min/max counts) for RL balance tracking
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import signal
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dyck_dataset import DyckDataset

# -----------------------------
# AMP helpers (speed)
# -----------------------------
def _get_amp_dtype(name: str):
    name = (name or "fp16").lower()
    if name in ("fp16", "float16", "16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown --amp_dtype {name}. Use fp16 or bf16.")

def _make_grad_scaler(enabled: bool):
    # torch.amp.GradScaler is preferred; fall back to torch.cuda.amp.GradScaler
    if not enabled:
        return None

    # Prefer the non-deprecated API when available.
    # PyTorch 2.x: torch.amp.GradScaler(device_type)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            # Some builds expose torch.amp.GradScaler without device_type arg
            return torch.amp.GradScaler()

    # Legacy fallback (may emit a deprecation warning on newer torch)
    return torch.cuda.amp.GradScaler()

from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

# ---- Local imports ----
from hypernet_model import create_hypernet_gpt2

# Dataset helpers
import importlib.util as _importlib_util

SCRIPT_DIR = Path(__file__).resolve().parent
_DATASET_IMPORT_ERROR = None
WikiTextDataset = None  # type: ignore
SimpleTextDataset = None  # type: ignore
HAS_DATASET = False

# ============================================================================
# GLOBAL INTERRUPT HANDLING
# ============================================================================
training_interrupted = False

def signal_handler(signum, frame):
    global training_interrupted
    print("\n⚠️  Training interruption requested...")
    print("Will save checkpoint and exit after current batch...")
    training_interrupted = True

signal.signal(signal.SIGINT, signal_handler)

# ============================================================================
# DATASET LOADING
# ============================================================================
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
        from dataset import WikiTextDataset as _WTD, SimpleTextDataset as _STD  # type: ignore
        WikiTextDataset, SimpleTextDataset = _WTD, _STD
        HAS_DATASET = True
except Exception as e:
    _DATASET_IMPORT_ERROR = e
    HAS_DATASET = False

# Orthogonal projection optimizer (optional)
OrthogonalProjectedAdamW = None
try:
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

def _single_token_ids(tokenizer, s: str):
    """
    Return a set of token ids that can represent this symbol in GPT-2 BPE.
    We try both 's' and ' '+s (space-prefixed) because your Dyck text is space-separated.
    If BPE splits, we keep the last token id as a fallback (still works reasonably).
    """
    out = set()
    for form in (s, " " + s):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            out.add(ids[0])
        elif len(ids) > 1:
            out.add(ids[-1])
    return out


def build_dyck_id_maps(tokenizer, opens, closes):
    # symbol -> set(token_ids)
    open_ids = {s: _single_token_ids(tokenizer, s) for s in opens}
    close_ids = {s: _single_token_ids(tokenizer, s) for s in closes}

    # token_id -> type_index (0/1/2/3)
    open_tid_to_type = {}
    close_tid_to_type = {}
    for i, s in enumerate(opens):
        for tid in open_ids[s]:
            open_tid_to_type[tid] = i
    for i, s in enumerate(closes):
        for tid in close_ids[s]:
            close_tid_to_type[tid] = i

    return open_tid_to_type, close_tid_to_type


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: GPT2Tokenizer,
        device: torch.device,
        lr: float,
        weight_decay: float,
        centroid_lr_mult: float = 0.2,
        hypernet_lr_mult: float = 10.0,
        use_orthogonal_projection: bool = False,
        projection_mode: str = "default",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Track rejuvenation events so eval can report whether revived centroids start winning.
        self.last_rejuvenated: List[int] = []

        # AMP (mixed precision) settings (configured in main() to keep Trainer signature unchanged)
        self.use_amp = False
        self.amp_dtype = torch.float16
        self.scaler = None

        # Track rejuvenated centroids so we can report whether they get used later
        self.last_rejuvenated_indices: List[int] = []

       # Split params: (1) everything except centroids, (2) centroids-like parameters
        # Split params: (1) base model, (2) hypernet, (3) centroids
        base_params = []
        hypernet_params = []
        centroid_params = []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n.lower() for k in ["centroid", "centroids"]):
                centroid_params.append(p)
            elif any(k in n.lower() for k in ["hypernet", "alpha_hypernet", "reasoning_injection"]):
                hypernet_params.append(p)
            else:
                base_params.append(p)

        # Check if orthogonal projection optimizer is available
        if use_orthogonal_projection and OrthogonalProjectedAdamW is None:
            print("⚠️  Orthogonal projection optimizer not found. Using standard AdamW for centroids.")

        def _make_optimizer(params, learning_rate=None, use_orthogonal=False):
            lr_to_use = learning_rate if learning_rate is not None else lr
            
            if use_orthogonal and OrthogonalProjectedAdamW is not None:
                try:
                    return OrthogonalProjectedAdamW(params, lr=lr_to_use, weight_decay=weight_decay, projection_mode=projection_mode)
                except TypeError:
                    return OrthogonalProjectedAdamW(params, lr=lr_to_use, weight_decay=weight_decay)
            
            return torch.optim.AdamW(params, lr=lr_to_use, weight_decay=weight_decay)

        # Separate learning rates
        hypernet_lr = lr * float(hypernet_lr_mult)
        centroid_lr = lr * float(centroid_lr_mult)

        self.main_optimizer = _make_optimizer(base_params)
        self.hypernet_optimizer = _make_optimizer(hypernet_params, learning_rate=hypernet_lr) if hypernet_params else None
        self.centroid_optimizer = _make_optimizer(
            centroid_params,
            learning_rate=centroid_lr,
            use_orthogonal=use_orthogonal_projection,
        ) if centroid_params else None

        self.main_scheduler = None
        self.hypernet_scheduler = None
        self.centroid_scheduler = None

    def attach_schedulers(self, total_steps: int, warmup_steps: int):
        self.main_scheduler = get_linear_schedule_with_warmup(
            self.main_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        if self.centroid_optimizer is not None:
            self.centroid_scheduler = get_linear_schedule_with_warmup(
                self.centroid_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        if self.hypernet_optimizer is not None:
            self.hypernet_scheduler = get_linear_schedule_with_warmup(
                self.hypernet_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        
        # Flag to disable NaN checking after training is stable
        self.training_started_successfully = False
        self.nan_check_until_step = 1000

    def train_step(self, batch: Dict[str, torch.Tensor], global_step: int = 0) -> Dict[str, float]:
        self.model.train()

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            # Ensure attention_mask is 2-D: [B, S]
            while attention_mask.dim() > 2:
                attention_mask = attention_mask.squeeze(1)
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask != 0

        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(~attention_mask, -100)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            loss = outputs.loss
    
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  NaN/Inf loss detected! Skipping batch...")
            # Clean up
            del outputs, loss, input_ids, labels
            if attention_mask is not None:
                del attention_mask
            torch.cuda.empty_cache()
            return {
                'lm_loss': float('nan'),
                'total_loss': float('nan'),
                'perplexity': float('nan'),
                'accuracy': 0.0,
                'gain': 0.0, 
            }

        self.main_optimizer.zero_grad(set_to_none=True)
        if self.hypernet_optimizer is not None:
            self.hypernet_optimizer.zero_grad(set_to_none=True)
        if self.centroid_optimizer is not None:
            self.centroid_optimizer.zero_grad(set_to_none=True)

        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Determine whether optimizers received grads (important for AMP)
        hypernet_has_grads = False
        if self.hypernet_optimizer is not None:
            for _g in self.hypernet_optimizer.param_groups:
                for _p in _g.get('params', []):
                    if _p is not None and _p.grad is not None:
                        hypernet_has_grads = True
                        break
                if hypernet_has_grads:
                    break
        
        centroid_has_grads = False
        if self.centroid_optimizer is not None:
            for _g in self.centroid_optimizer.param_groups:
                for _p in _g.get('params', []):
                    if _p is not None and _p.grad is not None:
                        centroid_has_grads = True
                        break
                if centroid_has_grads:
                    break
        
        # If using AMP, unscale gradients before any NaN/Inf checks or clipping.
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.main_optimizer)
            if self.hypernet_optimizer is not None and hypernet_has_grads:
                self.scaler.unscale_(self.hypernet_optimizer)
            if self.centroid_optimizer is not None and centroid_has_grads:
                self.scaler.unscale_(self.centroid_optimizer)

        # Check for NaN/Inf gradients (only in first N steps for efficiency)
        has_nan = False
        if global_step < self.nan_check_until_step:
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan = True
                        print(f"⚠️  NaN/Inf gradients detected at step {global_step}! Skipping update...")
                        break

        if has_nan:
            self.main_optimizer.zero_grad(set_to_none=True)
            if self.hypernet_optimizer is not None:
                self.hypernet_optimizer.zero_grad(set_to_none=True)
            if self.centroid_optimizer is not None:
                self.centroid_optimizer.zero_grad(set_to_none=True)
            # Let GradScaler react (reduce scale) even though we skip the step.
            if self.use_amp and self.scaler is not None:
                self.scaler.update()

            # Clean up
            del outputs, loss, input_ids, labels
            if attention_mask is not None:
                del attention_mask
            torch.cuda.empty_cache()
            return {
                'lm_loss': float('nan'),
                'total_loss': float('nan'),
                'perplexity': float('nan'),
                'accuracy': 0.0,
                'gain': 0.0,  
            }

        # Clip gradients more aggressively
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.main_optimizer)
        else:
            self.main_optimizer.step()
        if self.hypernet_optimizer is not None and hypernet_has_grads:
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.hypernet_optimizer)
            else:
                self.hypernet_optimizer.step()
        if self.centroid_optimizer is not None and centroid_has_grads:
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.centroid_optimizer)
            else:
                self.centroid_optimizer.step()

        if self.use_amp and self.scaler is not None:
            self.scaler.update()

        if self.main_scheduler is not None:
            self.main_scheduler.step()
        if self.hypernet_scheduler is not None and hypernet_has_grads:
            self.hypernet_scheduler.step()
        if self.centroid_scheduler is not None and centroid_has_grads:
            self.centroid_scheduler.step()

        lm_loss = float(loss.detach().item())

        # Calculate accuracy - more memory efficient
        with torch.no_grad():
            # Use detached tensors to avoid keeping computation graph
            logits = outputs.logits.detach()
            predictions = logits.argmax(dim=-1)
            # Shift for next-token prediction
            shift_preds = predictions[..., :-1].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Mask out padding tokens (-100)
            mask = shift_labels != -100
            correct = (shift_preds == shift_labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
            accuracy_value = float(accuracy.item() * 100)
            
            # Explicitly delete large tensors
            del logits, predictions, shift_preds, shift_labels, mask, correct, accuracy

        # Clean up
        del outputs, loss, input_ids, labels
        if attention_mask is not None:
            del attention_mask

        # Get current gain value from hypernet
        if hasattr(self.model, "hypernet") and self.model.hypernet is not None:
            gain_value = self.model.hypernet.min_gain + F.softplus(self.model.hypernet.hyper_gain_raw)
        else:
            gain_value = 0.0
        
        return {
            "lm_loss": lm_loss,
            "perplexity": float(math.exp(min(lm_loss, 20.0))),
            "accuracy": accuracy_value,
            "gain": gain_value,  
        }

    @torch.no_grad()
    def rejuvenate_dead_centroids(
        self,
        mix: float = 0.5,
        noise_std: float = 0.01,
    ) -> List[int]:
        """Rejuvenate dead centroids by mixing with active ones"""
        if not hasattr(self.model, "hypernet"):
            return []
        hyper = self.model.hypernet

        if not hasattr(hyper, "centroids"):
            print("[rejuvenate] model.hypernet has no .centroids")
            return []
        if not hasattr(hyper, "centroid_usage_counts"):
            print("[rejuvenate] model.hypernet has no .centroid_usage_counts")
            return []

        C = hyper.centroids
        counts = hyper.centroid_usage_counts

        dead = (counts == 0).nonzero(as_tuple=False).flatten()
        if dead.numel() == 0:
            return []

        alive = (counts > 0).nonzero(as_tuple=False).flatten()
        if alive.numel() == 0:
            return []

        m = dead.numel()

        # sort alive by count descending: biggest winners first
        alive_counts = counts[alive]
        alive_sorted = alive[torch.argsort(alive_counts, descending=True)]

        # how many dead centroids to revive via donors vs random restart
        m_from_donors = int(round(m * (2.0 / 3.0)))
        m_random = m - m_from_donors

        donor_k = min(m_from_donors, alive_sorted.numel())
        donors = alive_sorted[:donor_k]
        num_fixed = 0 

        revived: List[int] = []

        # ---- A) revive 2/3 by copying from top donors + mute donor slightly
        for idx in range(m_from_donors):
            j = int(dead[idx].item())
            i = int(donors[idx % donor_k].item())

            # Copy donor direction into dead centroid + small noise
            C[j].copy_(C[i])
            if noise_std > 0:
                C[j].add_(torch.randn_like(C[j]) * noise_std)

            # "Mute" the donor a little so it doesn't keep dominating forever
            mute_std = noise_std * 0.25
            if mute_std > 0:
                C[i].add_(torch.randn_like(C[i]) * mute_std)

            num_fixed += 1
            revived.append(j)

        # ---- B) restart 1/3 from random (exploration)
        for idx in range(m_from_donors, m):
            j = int(dead[idx].item())

            # Random restart then normalize-ish scale
            C[j].copy_(torch.randn_like(C[j]))
            C[j].mul_(0.02)  # keep small like typical init; adjust if needed
            if noise_std > 0:
                C[j].add_(torch.randn_like(C[j]) * (noise_std * 0.5))

            num_fixed += 1
            revived.append(j)

        # reset counts to re-measure after rebalance
        counts.zero_()
        # Record for later eval reporting
        self.last_rejuvenated = revived
        print(f"[rejuvenate] revived {num_fixed} dead centroids: "
            f"{m_from_donors} from donors (muted), {m_random} random restarts (counts reset)")
        return revived

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set with PPL and accuracy calculation
        
        Returns:
            Dictionary with 'ppl' (perplexity), 'acc' (accuracy), and 'nll' (negative log likelihood)
        """
        self.model.eval()

        # --- Dyck-specific evaluator (optional) ---
        is_dyck = hasattr(dataloader.dataset, "opens") and hasattr(dataloader.dataset, "closes")
        if is_dyck:
            opens = list(dataloader.dataset.opens)
            closes = list(dataloader.dataset.closes)
            open_tid_to_type, close_tid_to_type = build_dyck_id_maps(self.tokenizer, opens, closes)

            dyck_valid_total = 0
            dyck_valid_correct = 0
            dyck_close_total = 0
            dyck_close_correct = 0
        
        total_nll = 0.0
        total_tokens = 0
        total_correct = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device, non_blocking=True)
                # Ensure attention_mask is 2-D: [B, S]
                while attention_mask.dim() > 2:
                    attention_mask = attention_mask.squeeze(1)
                if attention_mask.dtype != torch.bool:
                    attention_mask = attention_mask != 0
            
            # Create labels
            labels = input_ids.clone()
            if attention_mask is not None:
                labels = labels.masked_fill(~attention_mask, -100)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            
            # Manual NLL and accuracy calculation
            logits = outputs.logits[..., :-1, :].contiguous()  # [B, S-1, V]
            targets = labels[..., 1:].contiguous()              # [B, S-1]
            valid_mask = (targets != -100)                      # [B, S-1]
            
            if valid_mask.any():
                flat_logits = logits[valid_mask]    # [N, V]
                flat_targets = targets[valid_mask]  # [N]
                
                # Log-probs for NLL
                log_probs = F.log_softmax(flat_logits, dim=-1)
                nll = -log_probs.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
                
                # Accuracy (argmax over logits)
                preds = flat_logits.argmax(dim=-1)  # [N]

                # --- Dyck metrics: "valid-next" and "close correctness" ---
                if is_dyck:
                    # We need per-position info (not flattened) to simulate the stack.
                    # logits is [B,S-1,V], targets is [B,S-1], valid_mask is [B,S-1]
                    pred_next = logits.argmax(dim=-1)  # [B,S-1]
                    max_depth = getattr(dataloader.dataset, "max_depth", None)

                    B, T = targets.shape  # T = S-1
                    for b in range(B):
                        stack = []
                        for t in range(T):
                            if not bool(valid_mask[b, t].item()):
                                break

                            # Current token in the context (teacher forced) is input_ids[b, t]
                            cur_tid = int(input_ids[b, t].item())

                            # Update stack based on CURRENT token
                            if cur_tid in open_tid_to_type:
                                stack.append(open_tid_to_type[cur_tid])
                            elif cur_tid in close_tid_to_type:
                                typ = close_tid_to_type[cur_tid]
                                if stack and stack[-1] == typ:
                                    stack.pop()
                                else:
                                    # Gold should be valid; ignore inconsistencies
                                    stack = []

                            # Determine valid NEXT symbols under Dyck rules
                            valid_next_types = set(open_tid_to_type.values())  # opens always allowed (depth cap ignored here)
                            valid_close_type = stack[-1] if stack else None

                            pred_tid = int(pred_next[b, t].item())
                            gold_tid = int(targets[b, t].item())

                            # "valid-next": either an open, or the correct close if stack non-empty
                            dyck_valid_total += 1
                            allow_open = True if max_depth is None else (len(stack) < int(max_depth))   
                            pred_is_open = allow_open and (pred_tid in open_tid_to_type)
                            pred_is_correct_close = (valid_close_type is not None and pred_tid in close_tid_to_type and close_tid_to_type[pred_tid] == valid_close_type)
                            if pred_is_open or pred_is_correct_close:
                                dyck_valid_correct += 1

                            # "close correctness": only score when GOLD next token is a close
                            if gold_tid in close_tid_to_type:
                                dyck_close_total += 1
                                if pred_tid == gold_tid:
                                    dyck_close_correct += 1

                    del pred_next

                total_correct += (preds == flat_targets).sum().item()
                
                # Accumulate
                total_nll += nll.sum().item()
                total_tokens += flat_targets.numel()
                
                # Clean up to prevent memory accumulation
                del flat_logits, flat_targets, log_probs, nll, preds
            
            # Clean up batch tensors
            del outputs, logits, targets, valid_mask, input_ids, labels
            if attention_mask is not None:
                del attention_mask
        
        if total_tokens == 0:
            print("[EVAL] No valid tokens found")
            return {"ppl": float('inf'), "acc": 0.0, "nll": float('inf')}
        
        avg_nll = total_nll / total_tokens
        ppl = math.exp(avg_nll)
        acc = total_correct / total_tokens  # fraction in [0,1]
        
        print(f"[EVAL] tokens={total_tokens} | nll={avg_nll:.6f} | ppl={ppl:.4f} | acc={acc*100:.2f}%")

        if is_dyck:
            dyck_valid_rate = dyck_valid_correct / max(1, dyck_valid_total)
            dyck_close_acc = dyck_close_correct / max(1, dyck_close_total)
            print(f"[DYCK] valid_next={dyck_valid_rate*100:.2f}% | close_acc={dyck_close_acc*100:.2f}% | close_total={dyck_close_total}")
        
        return {"ppl": ppl, "acc": acc, "nll": avg_nll}

    @torch.no_grad()
    def quick_preview_sample(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: Optional[float] = 0.9,
        top_k: int = 40,
    ) -> str:
        """Generate text sample from prompt for qualitative evaluation"""
        self.model.eval()
        
        # Encode prompt
        ctx = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ctx:
            ctx = [self.tokenizer.eos_token_id]
        input_ids = torch.tensor([ctx], device=self.device, dtype=torch.long)
        
        new_tokens = []
        for _ in range(max_new_tokens):
            attn = torch.ones_like(input_ids, device=self.device)
            
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                return_dict=True,
            )
            
            logits = out.logits[:, -1, :].float()
            
            # Sampling logic
            if temperature <= 0.0:
                next_id = int(torch.argmax(logits, dim=-1))
            else:
                l = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    k = min(top_k, l.size(-1))
                    kth = torch.topk(l, k=k, dim=-1).values[..., -1, None]
                    l = torch.where(l < kth, torch.full_like(l, float("-inf")), l)
                
                # Top-p filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    probs = torch.softmax(l, dim=-1)
                    sp, si = torch.sort(probs, descending=True)
                    cdf = torch.cumsum(sp, dim=-1)
                    keep = (cdf <= top_p)
                    keep[..., 0] = True
                    mask = torch.full_like(l, float("-inf"))
                    mask.scatter_(1, si, torch.where(keep, torch.zeros_like(sp), torch.full_like(sp, float("-inf"))))
                    l = l + mask
                
                p = torch.softmax(l, dim=-1)
                next_id = int(torch.multinomial(p, 1)[0, 0])
            
            new_tokens.append(next_id)
            
            # Check for EOS
            if next_id == self.tokenizer.eos_token_id:
                break
            
            # Append to input_ids for next iteration
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=self.device)], dim=1)
        
        # Decode generated tokens
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return generated

    def _probe_first_mlp_weights(self):
        """Probe first MLP layer weights for debugging"""
        try:
            # Try to access first layer MLP weights
            if hasattr(self.model, 'h') and len(self.model.h) > 0:
                first_block = self.model.h[0]
                if hasattr(first_block, 'mlp'):
                    mlp = first_block.mlp
                    # Print some weight statistics
                    for name, param in mlp.named_parameters():
                        if 'weight' in name:
                            print(f"[MLP-0] {name}: mean={param.mean().item():.4e} std={param.std().item():.4e}")
                            break  # Just show first weight
        except Exception as e:
            print(f"[MLP probe] Could not access MLP weights: {e}")

    @torch.no_grad()
    def monitor_centroid_wins(self):
        """
        Monitor centroid win counts (min/max) for balance tracking.
        Works with both RL-based and similarity-based centroid selection.
        Shows if some centroids are never winning (unbalanced training).
        """
        try:
            # Access hypernet blocks (layers with hypernetwork)
            if not hasattr(self.model, 'h'):
                return
            
            all_centroid_stats = []
            
            for layer_idx, block in enumerate(self.model.h):
                # Check if this block has a hypernetwork
                if not hasattr(block, 'hypernet_block'):
                    continue
                    
                hypernet = block.hypernet_block
                if not hasattr(hypernet, 'hypernetwork'):
                    continue
                
                hypernetwork = hypernet.hypernetwork
                num_centroids = hypernetwork.num_centroids
                
                # METHOD 1: Try RL trajectory buffer first
                centroid_counts = None
                source = None
                
                if hasattr(hypernetwork, 'trajectory_buffer') and hasattr(hypernetwork, 'use_rl') and hypernetwork.use_rl:
                    actions = hypernetwork.trajectory_buffer.get('actions', [])
                    
                    if actions:
                        # Count wins per centroid from RL actions
                        centroid_counts = torch.zeros(num_centroids, dtype=torch.long)
                        
                        for action_batch in actions:
                            if isinstance(action_batch, torch.Tensor):
                                flat_actions = action_batch.flatten()
                                for action in flat_actions:
                                    if 0 <= action < num_centroids:
                                        centroid_counts[action] += 1
                        source = "RL"
                
                # METHOD 2: If RL not available, check centroid parameters directly
                if centroid_counts is None:
                    # Analyze centroid usage by checking gradient activity
                    centroids = hypernetwork.centroids  # [num_centroids, hidden_size]
                    
                    # Check if centroids have been updated (gradient exists)
                    if centroids.grad is not None:
                        # Use gradient magnitude as proxy for usage
                        grad_norms = centroids.grad.norm(dim=1)  # [num_centroids]
                        
                        # Centroids with larger gradients are being used more
                        # Threshold: consider centroid "used" if grad > mean/10
                        threshold = grad_norms.mean() / 10
                        centroid_counts = (grad_norms > threshold).long()
                        source = "grad"
                    else:
                        # Fallback: check centroid weight norms
                        # Centroids that stay near initialization are unused
                        centroid_norms = centroids.norm(dim=1)  # [num_centroids]
                        mean_norm = centroid_norms.mean()
                        std_norm = centroid_norms.std()
                        
                        # Simple heuristic: show norm distribution
                        all_centroid_stats.append({
                            'layer': layer_idx,
                            'type': 'norms',
                            'min': centroid_norms.min().item(),
                            'max': centroid_norms.max().item(),
                            'mean': mean_norm.item(),
                            'std': std_norm.item(),
                        })
                        continue
                
                # Store statistics if we have counts
                if centroid_counts is not None:
                    all_centroid_stats.append({
                        'layer': layer_idx,
                        'counts': centroid_counts,
                        'source': source,
                    })
            
            # Print statistics
            if all_centroid_stats:
                print(f"\n🎯 Centroid Win Monitoring:")
                for item in all_centroid_stats:
                    layer_idx = item['layer']
                    
                    if item.get('type') == 'norms':
                        # Just show norm statistics
                        print(f"   Layer {layer_idx} (norm-based): "
                              f"min={item['min']:.3f} | max={item['max']:.3f} | "
                              f"mean={item['mean']:.3f} | std={item['std']:.3f}")
                    else:
                        # Show count statistics
                        counts = item['counts']
                        source = item.get('source', 'unknown')
                        
                        min_count = counts.min().item()
                        max_count = counts.max().item()
                        mean_count = counts.float().mean().item()
                        num_zeros = (counts == 0).sum().item()
                        
                        print(f"   Layer {layer_idx} ({source}): "
                              f"min={min_count:4d} | max={max_count:4d} | mean={mean_count:6.1f} | "
                              f"unused={num_zeros}/{len(counts)}")
                        
                        if num_zeros > 0:
                            print(f"      ⚠️  {num_zeros} centroids never won!")
                print()
            else:
                # No statistics available - provide helpful message
                print(f"[Centroid Monitor] No centroid usage data available yet")
                print(f"   Tip: Centroid monitoring works best with:")
                print(f"   - RL enabled (use_rl=True) OR")
                print(f"   - After several training steps (gradients accumulate)")
                    
        except Exception as e:
            print(f"[Centroid Monitor] Error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# CHECKPOINT SAVE/LOAD FUNCTIONS
# ============================================================================

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    trainer: Trainer,
    epoch: int,
    global_step: int,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save checkpoint with model, optimizer, scheduler states
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        trainer: Trainer object with optimizers and schedulers
        epoch: Current epoch
        global_step: Current global step
        extra: Optional extra data to save (e.g., args)
        hypernet_optimizer: hypernet optimizer
        hypernet_scheduler: hypernet scheduler
    """
    payload = {
        "model_state": model.state_dict(),
        "main_optimizer": trainer.main_optimizer.state_dict(),
        "centroid_optimizer": trainer.centroid_optimizer.state_dict() if trainer.centroid_optimizer else None,
        "main_scheduler": trainer.main_scheduler.state_dict() if trainer.main_scheduler else None,
        "centroid_scheduler": trainer.centroid_scheduler.state_dict() if trainer.centroid_scheduler else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "extra": extra or {},
        "hypernet_optimizer": trainer.hypernet_optimizer.state_dict() if trainer.hypernet_optimizer else None,
        "hypernet_scheduler": trainer.hypernet_scheduler.state_dict() if trainer.hypernet_scheduler else None,
    }
    torch.save(payload, str(path))
    print(f"💾 Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    trainer: Trainer
) -> Tuple[int, int]:
    """
    Load checkpoint and restore model, optimizer, scheduler states
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        trainer: Trainer object with optimizers and schedulers
    
    Returns:
        Tuple of (epoch, global_step) from checkpoint
    """
    print(f"🔄 Loading checkpoint: {path}")
    ckpt = torch.load(str(path), map_location="cpu")
    
    # Load model state
    model.load_state_dict(ckpt["model_state"], strict=True)
    
    # Load optimizer states (only if they exist)
    if "main_optimizer" in ckpt and ckpt["main_optimizer"] is not None:
        trainer.main_optimizer.load_state_dict(ckpt["main_optimizer"])
        print("  ✓ Loaded main_optimizer state")
    else:
        print("  ⚠️  No main_optimizer in checkpoint (starting fresh)")
    if trainer.main_scheduler is not None and ckpt.get("main_scheduler") is not None:
        trainer.main_scheduler.load_state_dict(ckpt["main_scheduler"])
    
    if trainer.centroid_optimizer is not None and ckpt.get("centroid_optimizer") is not None:
        trainer.centroid_optimizer.load_state_dict(ckpt["centroid_optimizer"])
    if trainer.centroid_scheduler is not None and ckpt.get("centroid_scheduler") is not None:
        trainer.centroid_scheduler.load_state_dict(ckpt["centroid_scheduler"])
    
    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))

    if "hypernet_optimizer" in ckpt and ckpt["hypernet_optimizer"] is not None and trainer.hypernet_optimizer is not None:
        trainer.hypernet_optimizer.load_state_dict(ckpt["hypernet_optimizer"])
    if "hypernet_scheduler" in ckpt and ckpt["hypernet_scheduler"] is not None and trainer.hypernet_scheduler is not None:
        trainer.hypernet_scheduler.load_state_dict(ckpt["hypernet_scheduler"])
    
    print(f"✅ Checkpoint loaded: epoch={epoch}, global_step={global_step}")
    return epoch, global_step


# ============================================================================
# MODEL AND DATASET BUILDERS
# ============================================================================

def build_model(args, tokenizer: GPT2Tokenizer, device: torch.device) -> torch.nn.Module:
    result = create_hypernet_gpt2(
        base_model=args.base_model,
        use_pretrained=args.use_pretrained,
        num_centroids=args.num_centroids,
        memory_window=args.memory_window,
        reasoning_json_path=args.reasoning_json_path,
        use_rl=args.use_rl,  # Use the argument instead of hardcoded False
        rl_method=args.rl_method,
        temperature=args.temperature, 
        top_k_train=args.top_k_train,
        top_k_eval=args.top_k_eval,
        balance_weight=args.balance_weight,
        balance_mode=args.balance_mode,
    )

    # Robust: accept either model OR (model, config)
    if isinstance(result, tuple) and len(result) >= 1:
        model = result[0]
    else:
        model = result

    if hasattr(model, "set_tokenizer"):
        model.set_tokenizer(tokenizer)

    # Propagate disable_hypernet flag to the hypernet module
    if getattr(args, "disable_hypernet", False) and hasattr(model, "hypernet"):
        model.hypernet.disable_hypernet = True
        print("⚠️ HyperNet DISABLED — reasoning_vector will be None")

    model = model.to(device)
    return model


def build_dataset(args, tokenizer: GPT2Tokenizer, split: str = "train"):
    """Build dataset for training or validation"""
    if args.data_dir:
        if not HAS_DATASET or WikiTextDataset is None:
            raise RuntimeError(f"Could not load WikiTextDataset from local dataset.py. Import error: {_DATASET_IMPORT_ERROR}")
        return WikiTextDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            split=split,
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
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)

    # Dyck
    p.add_argument("--task", type=str, default="dyck", help="Train dyck dataset.")
    p.add_argument("--train_samples", type=int, default=200000)
    p.add_argument("--dyck_train_depth", type=int, default=20)
    p.add_argument("--dyck_eval_depth", type=int, default=100)
    p.add_argument("--dyck_types", type=int, default=2)
    p.add_argument("--disable_hypernet", action="store_true")

    p.add_argument("--dyck_eval_p_continue", type=float, default=0.7)
    p.add_argument("--dyck_eval_min_depth", type=int, default=0)
    p.add_argument("--dyck_eval_stop_prob", type=float, default=0.2)
   
    # Speed
    p.add_argument("--amp", type=_str2bool, default=True, help="Use AMP (mixed precision) for faster training on GPU.")
    p.add_argument("--amp_dtype", type=str, default="fp16", help="AMP dtype: fp16 or bf16.")
    p.add_argument("--allow_tf32", type=_str2bool, default=True, help="Allow TF32 matmul on Ampere+ GPUs for speed.")

    # Model
    p.add_argument("--base_model", type=str, default="gpt2")
    p.add_argument("--use_pretrained", action="store_true", help="Initialize from HF pretrained GPT-2 weights.")
    p.add_argument("--num_centroids", type=int, default=128)
    p.add_argument("--memory_window", type=int, default=10)
    p.add_argument("--reasoning_json_path", type=str, default="reasoning_logic.json")
    p.add_argument("--use_rl", type=_str2bool, default=False, 
                   help="Enable RL for centroid selection (default: True).")
    p.add_argument("--rl_method", type=str, default="actor_critic")

    p.add_argument(
        "--temperature",
        type=float,
        default=5.0,
        help="Temperature for centroid routing during training (default: 5.0)"
    )

    # Centroid routing + load balancing
    p.add_argument("--top_k_train", type=int, default=1, help="Top-k centroid mixture during training (1 = hard top-1)")
    p.add_argument("--top_k_eval", type=int, default=1, help="Top-k centroid mixture during eval (usually 1)")
    p.add_argument("--balance_weight", type=float, default=0.01, help="Weight for centroid load-balance loss (0 disables)")
    p.add_argument("--balance_mode", type=str, default="switch", choices=["switch", "l2_uniform", "both", "off"], help="Balance loss mode")

    # LR multipliers (separate optimizers)
    p.add_argument("--centroid_lr_mult", type=float, default=0.2, help="Centroid LR multiplier relative to --lr")
    p.add_argument("--hypernet_lr_mult", type=float, default=10.0, help="Non-centroid hypernet LR multiplier relative to --lr")
    p.add_argument("--rejuvenate_steps", type=int, default=500, help="rejuvenate centroid every N steps.")
   
    # Train
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=0, help="0 = use epochs")

    # Checkpointing
    p.add_argument("--output_dir", type=str, default="./out")
    p.add_argument("--save_every", type=int, default=200, help="Save checkpoint every N steps.")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Evaluation
    p.add_argument("--eval_every", type=int, default=500, help="Evaluate every N steps.")
    p.add_argument("--eval_samples", type=int, default=500, help="Number of samples for validation.")

    # Optimizer options
    p.add_argument("--use_orthogonal_projection", action="store_true")
    p.add_argument("--projection_mode", type=str, default="default")

    p.add_argument(
        "--eval_only",
        action="store_true",
        help="Run evaluation only and exit (no training)"
    )

    return p.parse_args()

def main():
    global training_interrupted
    
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # TF32 for speed on Ampere+ GPUs
    if device.type == 'cuda' and args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass


    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    print("📦 Building model...")
    model = build_model(args, tokenizer, device)

    # Dataset
    print("📚 Loading training dataset...")
    if args.task == "dyck":
        train_dataset = DyckDataset(
            tokenizer=tokenizer,
            max_length=args.max_length,
            n_samples=args.train_samples,
            max_depth=args.dyck_train_depth,
            dyck_types=args.dyck_types,
            p_continue=args.dyck_eval_p_continue,
            min_depth=args.dyck_eval_min_depth,
            stop_prob=args.dyck_eval_stop_prob,
            seed=1234,
        )
        val_dataset = DyckDataset(
            tokenizer=tokenizer,
            max_length=args.max_length,
            n_samples=args.eval_samples,
            max_depth=args.dyck_eval_depth,
            dyck_types=args.dyck_types,
            p_continue=args.dyck_eval_p_continue,
            min_depth=args.dyck_eval_min_depth,
            stop_prob=args.dyck_eval_stop_prob,
            seed=999,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"✅ Dyck datasets loaded: train={len(train_dataset)} | val={len(val_dataset)}")
    else:
        train_dataset = build_dataset(args, tokenizer, split="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"✅ Training dataset loaded: {len(train_dataset)} samples")

        # Validation dataset
        print("📚 Loading validation dataset...")
        try:
            val_dataset = build_dataset(args, tokenizer, split="valid")
            # Limit validation set size
            if len(val_dataset) > args.eval_samples:
                val_dataset.texts = val_dataset.texts[:args.eval_samples]
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            print(f"✅ Validation dataset loaded: {len(val_dataset)} samples")
        except Exception as e:
            print(f"⚠️  Could not load validation dataset: {e}")
            val_loader = None

    epoch_steps = len(train_loader)
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else epoch_steps * args.num_epochs
    )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        centroid_lr_mult=args.centroid_lr_mult,
        hypernet_lr_mult=args.hypernet_lr_mult,
        use_orthogonal_projection=args.use_orthogonal_projection,
        projection_mode=args.projection_mode,
    )
    trainer.attach_schedulers(total_steps=total_steps, warmup_steps=args.warmup_steps)

    # ------------------------------------------------------------------
    # AMP configuration (speed-only). Kept outside Trainer.__init__ so the
    # Trainer signature remains unchanged.
    # ------------------------------------------------------------------
    trainer.use_amp = bool(args.amp) and (device.type == "cuda")
    trainer.amp_dtype = _get_amp_dtype(args.amp_dtype) if trainer.use_amp else torch.float16
    trainer.scaler = _make_grad_scaler(trainer.use_amp and (trainer.amp_dtype == torch.float16))

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume or save initial checkpoint
    start_epoch = 0
    global_step = 0
    
    if args.resume_from_checkpoint:
        ckpt_path = Path(args.resume_from_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume_from_checkpoint not found: {ckpt_path}")
        start_epoch, global_step = load_checkpoint(ckpt_path, model, trainer)

        # Add this diagnostic after loading checkpoint
        with torch.no_grad():
            gain = model.hypernet.min_gain + F.softplus(model.hypernet.hyper_gain_raw)
            print(f"hyper_gain: {gain.item():.4f}")
            
            for i, block in enumerate(model.h):
                if hasattr(block, 'mlp_proj_extended'):
                    w = block.mlp_proj_extended.weight
                    base_norm = w[:, :3072].norm().item()
                    hyper_norm = w[:, 3072:].norm().item()
                    print(f"Layer {i}: base_norm={base_norm:.2f}, hyper_norm={hyper_norm:.4f}, ratio={hyper_norm/base_norm:.4f}")

        # ==========================================================
        # Eval-only mode (no training)
        # ==========================================================
        if args.eval_only:
            print("\n🧪 Running EVAL-ONLY mode")

            if val_loader is None:
                raise RuntimeError("eval_only requested but val_loader is None")

            # Ensure eval mode
            trainer.model.eval()

            print(f"\n{'='*60}")
            print(f"📊 EVALUATION (eval_only)")
            print(f"{'='*60}")

            # Dropout sanity check (optional)
            try:
                device_model = next(model.parameters()).device
                E = model.config.n_embd if hasattr(model, 'config') else 768
                x = torch.ones(2, 3, E, device=device_model)

                model.train()
                y_train = model.h[0].mlp_dropout(x)

                model.eval()
                with torch.no_grad():
                    y_eval = model.h[0].mlp_dropout(x)

                print("train==eval?", torch.allclose(y_train, y_eval))
                print(f"std train: {float(y_train.std().cpu()):.6f}, std eval: {float(y_eval.std().cpu()):.6f}")
            except Exception as e:
                print(f"[Dropout check skipped: {e}]")

            # Centroid usage snapshot
            if hasattr(trainer.model, "hypernet"):
                usage = model.hypernet.get_centroid_usage(topk=10, reset=False)
                print("Centroid usage total:", usage["total"])
                print("Top centroids:", usage["top"])

            # Core evaluation
            eval_metrics = trainer.evaluate(val_loader)

            print(f"{'='*60}\n")
            print("✅ Eval-only completed. Exiting.")
            return

    else:
        # Save initial checkpoint before training to test preload correctness
        print("\n💾 Saving initial checkpoint (step 0) to test preload correctness...")
        initial_ckpt = output_dir / "checkpoint-initial.pt"
        save_checkpoint(initial_ckpt, model, trainer, epoch=0, global_step=0, extra={"args": vars(args)})
        
        # Test loading the initial checkpoint
        print("🧪 Testing checkpoint load/save cycle...")
        test_epoch, test_step = load_checkpoint(initial_ckpt, model, trainer)
        assert test_epoch == 0 and test_step == 0, "Checkpoint load test failed!"
        print("✅ Checkpoint load/save test passed!")

    print("\n🚀 Training Configuration")
    print(f"   epochs: {args.num_epochs}")
    print(f"   steps/epoch: {len(train_loader)}")
    print(f"   total steps: {total_steps}")
    print(f"   save_every: {args.save_every}")
    print(f"   eval_every: {args.eval_every}")
    print(f"   output_dir: {output_dir}\n")

    # Training loop with interrupt handling
    epoch = start_epoch  # Initialize for finally block
    try:
        for epoch in range(start_epoch, args.num_epochs):
            if training_interrupted:
                break
                
            running_loss = 0.0
            running_acc = 0.0
            running_gain = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                if training_interrupted:
                    break
                    
                global_step += 1
                # --- max_steps enforcement ---
                if args.max_steps > 0 and global_step > args.max_steps:
                    print(f"\n🛑 Reached max_steps={args.max_steps}. Stopping training.")
                    
                    ckpt = output_dir / f"checkpoint-maxsteps-{args.max_steps}.pt"
                    save_checkpoint(
                        ckpt,
                        model,
                        trainer,
                        epoch=epoch,
                        global_step=args.max_steps,
                        extra={"args": vars(args)}
                    )
                    
                    training_interrupted = True
                    break

                metrics = trainer.train_step(batch, global_step=global_step)
                running_loss += metrics["lm_loss"]
                running_acc += metrics["accuracy"]
                running_gain += metrics["gain"]
                
                # Clear cache every 50 steps to prevent memory fragmentation
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()

                # Logging
                if global_step % 100 == 0:
                    avg = running_loss / 100.0
                    avg_acc = running_acc / 100.0
                    avg_gain = running_gain / 100.0
                    running_loss = 0.0
                    running_acc = 0.0
                    
                    
                    print(
                        f"epoch {epoch+1}/{args.num_epochs} | step {global_step}/{total_steps} "
                        f"| loss {avg:.4f} | ppl {math.exp(min(avg, 20.0)):.2f} | acc {avg_acc:.2f}% gain {avg_gain:.2f}"
                    )

                # Evaluation with PPL, accuracy, and generation sample
                if args.eval_every > 0 and global_step > 0 and (global_step % args.eval_every == 0) and val_loader is not None:
                    print(f"\n{'='*60}")
                    print(f"📊 EVALUATION at step {global_step}")
                    print(f"{'='*60}")
                    
                    # Check dropout behavior (train vs eval mode)
                    prev_mode = trainer.model.training
                    try:
                        device_model = next(model.parameters()).device
                        E = model.config.n_embd if hasattr(model, 'config') else 768
                        x = torch.ones(2, 3, E, device=device_model)
                        
                        model.train()
                        y_train = model.h[0].mlp_dropout(x) if hasattr(model, 'h') else x
                        
                        model.eval()
                        with torch.no_grad():
                            y_eval = model.h[0].mlp_dropout(x) if hasattr(model, 'h') else x
                        
                        print("train==eval?", torch.allclose(y_train, y_eval))
                        print(f"std train: {float(y_train.std().cpu()):.6f}, std eval: {float(y_eval.std().cpu()):.6f}")
                    except Exception as e:
                        print(f"[Dropout check skipped: {e}]")
                    
                    # Probe MLP weights
                    trainer._probe_first_mlp_weights()
                    
                    # Monitor centroid wins (NEW FEATURE!)
                    # trainer.monitor_centroid_wins()
                    if hasattr(trainer.model, "hypernet"):
                        usage = model.hypernet.get_centroid_usage(topk=10, reset=False)
                        print("Centroid usage total:", usage["total"])
                        print("Top centroids:", usage["top"])
                        if getattr(trainer, "last_rejuvenated", None):
                            revived = trainer.last_rejuvenated
                            total_counts = usage["total"]
                            revived_nonzero = [i for i in revived if i < len(total_counts) and int(total_counts[i]) > 0]
                            print(f"Revived centroids since last rejuvenate: {len(revived)} | now nonzero: {len(revived_nonzero)}")
                    
                    # Set to eval mode for evaluation
                    trainer.model.eval()
                    
                    # PPL and Accuracy evaluation
                    eval_metrics = trainer.evaluate(val_loader)
                    
                    # Clear RL trajectory buffer to save memory
                    if hasattr(trainer.model, 'hypernet') and hasattr(trainer.model.hypernet, 'clear_trajectory'):
                        trainer.model.hypernet.clear_trajectory()
                    
                    # Generation sample from first batch
                    try:
                        ctx_ids = batch["input_ids"][0][:32].detach().cpu()
                        prompt_text = trainer.tokenizer.decode(ctx_ids, skip_special_tokens=True)
                        
                        sample_text = trainer.quick_preview_sample(
                            prompt_text,
                            max_new_tokens=64,
                            temperature=0.8,
                            top_p=0.9,
                            top_k=40
                        )
                        
                        print(f"\n📝 Generation Sample:")
                        print(f"Prompt: '{prompt_text[:80]}...'")
                        print(f"Generated: '{sample_text}'")
                    except Exception as e:
                        print(f"[Generation sample failed: {e}]")
                    
                    # Restore training mode if it was on
                    if prev_mode:
                        trainer.model.train()
                    
                    print(f"{'='*60}\n")

                    # Clean up after evaluation to free memory
                    torch.cuda.empty_cache()

                if args.rejuvenate_steps > 0 and (global_step % args.rejuvenate_steps == 0):
                    trainer.rejuvenate_dead_centroids()

                # Save checkpoint
                if args.save_every > 0 and (global_step % args.save_every == 0):
                    ckpt = output_dir / f"checkpoint-{global_step}.pt"
                    save_checkpoint(ckpt, model, trainer, epoch=epoch, global_step=global_step, extra={"args": vars(args)})

            # Epoch-end checkpoint (only if not interrupted)
            if not training_interrupted:
                ckpt = output_dir / f"checkpoint-epoch{epoch+1}-step{global_step}.pt"
                save_checkpoint(ckpt, model, trainer, epoch=epoch+1, global_step=global_step, extra={"args": vars(args)})

        if not training_interrupted:
            print("✅ Training completed!")
        else:
            print("⚠️  Training interrupted by user")
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by KeyboardInterrupt")
        training_interrupted = True
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        training_interrupted = True
    finally:
        # Save final checkpoint if interrupted
        if training_interrupted:
            print(f"\n💾 Saving interrupt checkpoint at step {global_step}...")
            interrupt_ckpt = output_dir / f"checkpoint-interrupted-step{global_step}.pt"
            try:
                save_checkpoint(interrupt_ckpt, model, trainer, epoch=epoch, global_step=global_step, extra={"args": vars(args)})
                print(f"✅ Interrupt checkpoint saved: {interrupt_ckpt}")
            except Exception as e:
                print(f"❌ Failed to save interrupt checkpoint: {e}")
        
        print(f"\n📁 All files saved to: {output_dir}")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()