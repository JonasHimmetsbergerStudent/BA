"""Auto-regressive GPT model trained on binned spike data.

Based on https://github.com/guillaumeBellec/tiny-gpt/blob/main/main.py
but adapted to predict spike counts instead of text tokens.

Task system: a list of TaskSpec defines all tasks. Each training step picks
one available task from the current data chunk. Per-task prediction heads
live in the adapter; per-task embeddings tell the model which signal to predict.

Loss: binary cross-entropy for spikes, MSE for regression, CE for classification.
Validation: last 2 minutes of drifting gratings per session.

Batching: each step samples B distant windows from the same session, sharing
the adapter (w_in/w_out/b_out). The transformer forward pass is batched.
"""

"""
Changes made so far:
- Added dropout to the Block, MLP class
- Added coordinate-based output weights for the adapter (Coordinate aware embedding)
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gradient_normalizer import GradientNormalizer
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)


# ---------------------------------------------------------------------------
# Transformer components (from tiny-gpt)
# ---------------------------------------------------------------------------

class CastedLinear(nn.Linear):
    """Linear layer that casts weights to input dtype (keeps master weights in float32)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, positions=None):
        if positions is None:
            positions = torch.arange(x.shape[1], device=x.device).float()
        freqs = torch.outer(positions.float(), self.inv_freq)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = CastedLinear(config.n_embd, config.n_embd, bias=False)
        self.c_k = CastedLinear(config.n_embd, config.n_embd, bias=False)
        self.c_v = CastedLinear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = CastedLinear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)

    def forward(self, x, block_mask, positions=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q, positions)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = CastedLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = CastedLinear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # return self.c_proj(F.relu(self.c_fc(x)).square())
        hidden = F.relu(self.c_fc(x)).square()
        hidden = self.dropout(hidden)
        return self.c_proj(hidden)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, block_mask, positions=None):
        # x = x + self.attn(F.rms_norm(x, (x.size(-1),)), block_mask, positions)
        # x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        # return x
        attn_out = self.attn(F.rms_norm(x, (x.size(-1),)), block_mask, positions)
        x = x + self.resid_drop(attn_out)
        mlp_out = self.mlp(F.rms_norm(x, (x.size(-1),)))
        x = x + self.resid_drop(mlp_out)
        return x


class SpikeGPTConfig:
    def __init__(self, n_layer=8, n_head=6, n_embd=384, window_size=128, dropout=0.1):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.window_size = window_size  # sliding window for causal attention
        self.dropout = dropout

# Model configs (GPT-2 naming convention).
#
# Chinchilla-optimal scaling: ~20 tokens per parameter (Hoffmann et al., 2022,
# https://arxiv.org/abs/2203.15556). Here a "token" = one time-bin embedding.
# Relative data requirements (proportional to parameter count):
#   small needs  ~6x more data than tiny
#   medium needs ~4x more data than small (~21x more than tiny)
CONFIGS = {
    "tiny": SpikeGPTConfig(n_layer=8, n_head=6, n_embd=384, window_size=32, dropout=0.1),
    "small": SpikeGPTConfig(n_layer=12, n_head=12, n_embd=768, window_size=64, dropout=0.1),
    "medium": SpikeGPTConfig(n_layer=24, n_head=16, n_embd=1024, window_size=128, dropout=0.1),
}


class SpikeGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.sos_embed = nn.Parameter(torch.randn(config.n_embd) * 0.02)

    def forward(self, x, block_mask, positions=None):
        for block in self.blocks:
            x = block(x, block_mask, positions)
        return F.rms_norm(x, (x.size(-1),))


# ---------------------------------------------------------------------------
# Stochastic rounding ReLU
# ---------------------------------------------------------------------------

def stochastic_binary_spike(logits):
    """Bernoulli sampling from logits with straight-through gradient.
    Forward: sample ~ Bernoulli(sigmoid(logits))  →  binary {0, 1}.
    Backward: gradient of sigmoid(logits)  (straight-through past sampling)."""
    prob = torch.sigmoid(logits)
    sample = (torch.rand_like(prob) < prob).float()
    return prob + (sample - prob).detach()  # straight-through


# ---------------------------------------------------------------------------
# Task specification
#
# Three subclasses: ARTask, RegressionTask, ClassificationTask.
# Each owns its signal data and provides compute_loss / update_val_metrics.
# ---------------------------------------------------------------------------

class TaskSpec:
    """Base class for all tasks."""
    def __init__(self, name):
        self.name = name

    def load_session(self, session, binned_signals):
        pass

    def has_session(self, session):
        return True

    def is_available(self, session, t0, seq_len):
        return True

    def prepare(self, train_ends):
        """Called once after all sessions loaded (normalization, etc)."""
        pass

    def compute_loss(self, h, adapter, session, t0, seq_len, device,
                     z=None, visible_idx=None, hidden_idx=None):
        raise NotImplementedError

    def reset_val_metrics(self):
        pass

    def update_val_metrics(self, h, adapter, session, t0, seq_len, device,
                           z=None, visible_idx=None, hidden_idx=None):
        pass

    def get_val_metrics(self):
        """Return dict of metric_name -> value."""
        return {}


class ARTask(TaskSpec):
    """Auto-regressive spike prediction (visible or hidden units).

    Accepts h of shape (B, T, d) or (T, d), z of shape (B, T, N) or (T, N).
    """
    def __init__(self, name, target="visible"):
        super().__init__(name)
        self.target = target  # "visible" or "hidden"

    def compute_loss(self, h, adapter, session, t0s, seq_len, device,
                     z=None, visible_idx=None, hidden_idx=None):
        # h: (B, T, d), z: (B, T, N)
        logits = h @ adapter.w_out(session).T + adapter.b_out(session)  # (B, T, N)
        idx = visible_idx if self.target == "visible" else hidden_idx
        logits = logits[:, :-1, idx]       # (B, T-1, n_idx)
        target = (z[:, 1:, idx] > 0.5).float()
        return F.binary_cross_entropy_with_logits(logits, target)

    def reset_val_metrics(self):
        self._tp = self._fp = self._fn = 0

    def update_val_metrics(self, h, adapter, session, t0s, seq_len, device,
                           z=None, visible_idx=None, hidden_idx=None, dt=None):
        if self.target != "hidden":
            return
        logits = h @ adapter.w_out(session).T + adapter.b_out(session)
        # Sample full z_pred raster from Bernoulli at native dt
        probs = torch.sigmoid(logits[:, :-1, hidden_idx])
        pred_bin = torch.bernoulli(probs).bool()       # (B, T-1, N_hid)
        real_bin = z[:, 1:, hidden_idx] > 0.5           # (B, T-1, N_hid)
        # Rebin to 10ms bins for comparable F1 across dt values
        f1_bin = 0.01  # 10ms
        rebin = max(1, round(f1_bin / dt))
        if rebin > 1:
            B, T, N = pred_bin.shape
            T_trim = (T // rebin) * rebin
            pred_bin = pred_bin[:, :T_trim].reshape(B, T_trim // rebin, rebin, N).any(dim=2)
            real_bin = real_bin[:, :T_trim].reshape(B, T_trim // rebin, rebin, N).any(dim=2)
        self._tp += (pred_bin & real_bin).sum().item()
        self._fp += (pred_bin & ~real_bin).sum().item()
        self._fn += (~pred_bin & real_bin).sum().item()

    def get_val_metrics(self):
        if self.target != "hidden":
            return {}
        prec = self._tp / max(1, self._tp + self._fp)
        rec = self._tp / max(1, self._tp + self._fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        return {"spike_f1": f1}


class RegressionTask(TaskSpec):
    """Predict a continuous signal from spikes (scalar or multi-dim)."""
    def __init__(self, name, signal_key):
        super().__init__(name)
        self.signal_key = signal_key
        self.d_out = 1  # updated in load_session based on data shape
        self._data = {}  # session -> (T, d_s+1) array
        self._norm_mean = None
        self._norm_std = None

    def load_session(self, session, binned_signals):
        if self.signal_key in binned_signals:
            self._data[session] = binned_signals[self.signal_key]
            self.d_out = self._data[session].shape[1] - 1  # last col is validity

    def has_session(self, session):
        return session in self._data

    def is_available(self, session, t0, seq_len):
        if session not in self._data:
            return False
        return (self._data[session][t0:t0 + seq_len, -1] > 0.5).any()

    def prepare(self, train_ends):
        """Compute z-score stats from training data (applied lazily in _get_target_batch)."""
        total_sum = total_sum_sq = None
        total_n = 0
        for session, arr in self._data.items():
            if session not in train_ends:
                continue
            d_s = arr.shape[1] - 1
            if d_s == 0:
                continue
            train_arr = arr[:train_ends[session]]
            valid = train_arr[:, d_s] > 0.5
            if not valid.any():
                continue
            if total_sum is None:
                total_sum = np.zeros(d_s)
                total_sum_sq = np.zeros(d_s)
            total_sum += train_arr[valid, :d_s].sum(axis=0)
            total_sum_sq += (train_arr[valid, :d_s] ** 2).sum(axis=0)
            total_n += valid.sum()
        if total_n == 0 or total_sum is None:
            self._norm_mean = None
            self._norm_std = None
            return
        self._norm_mean = total_sum / total_n
        self._norm_std = np.sqrt(total_sum_sq / total_n - self._norm_mean ** 2).clip(min=1e-6)
        print(f"    {self.name}: normalized ({total_n} valid samples)")

    def _get_target_batch(self, session, t0s, seq_len, device):
        # t0s: list of B start positions
        arrs = [self._data[session][t0:t0 + seq_len] for t0 in t0s]
        arr = np.stack(arrs)  # (B, T, d_out+1)
        valid = torch.from_numpy(arr[:, :, -1] > 0.5).to(device)  # (B, T)
        raw = arr[:, :, :self.d_out]
        if self._norm_mean is not None:
            raw = (raw - self._norm_mean) / self._norm_std
        target = torch.from_numpy(
            np.ascontiguousarray(raw)
        ).float().to(device)  # (B, T, d_out)
        return target, valid

    def compute_loss(self, h, adapter, session, t0s, seq_len, device, **_):
        target, valid = self._get_target_batch(session, t0s, seq_len, device)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        head = adapter.task_heads[self.name]
        pred = h @ head.T  # (B, T, d_out)
        return F.mse_loss(pred[valid], target[valid])

    def reset_val_metrics(self):
        self._ss_res = self._sum = self._sum_sq = 0.0
        self._n = 0

    def update_val_metrics(self, h, adapter, session, t0s, seq_len, device, dt=None, **_):
        target, valid = self._get_target_batch(session, t0s, seq_len, device)
        if not valid.any():
            return
        head = adapter.task_heads[self.name]
        pred = h @ head.T  # (B, T, d_out)
        # Rebin to 100ms bins: average predictions and targets
        reg_bin = 0.1  # 100ms
        rebin = max(1, round(reg_bin / dt)) if dt else 1
        if rebin > 1:
            B, T, d_out = pred.shape
            T_trim = (T // rebin) * rebin
            pred = pred[:, :T_trim].reshape(B, T_trim // rebin, rebin, d_out).mean(dim=2)
            target = target[:, :T_trim].reshape(B, T_trim // rebin, rebin, d_out).mean(dim=2)
            valid = valid[:, :T_trim].reshape(B, T_trim // rebin, rebin).any(dim=2)
        tv = target[valid]  # (N_valid, d_out)
        pv = pred[valid]
        self._ss_res += ((pv - tv) ** 2).sum().item()
        self._sum += tv.sum().item()
        self._sum_sq += (tv ** 2).sum().item()
        self._n += tv.numel()

    def get_val_metrics(self):
        if self._n < 2:
            return {}
        mean = self._sum / self._n
        ss_tot = self._sum_sq - self._n * mean ** 2
        r2 = 1.0 - self._ss_res / max(ss_tot, 1e-8)
        return {f"r2/{self.name}": r2}


class ClassificationTask(TaskSpec):
    """Predict a discrete condition index from spikes."""
    def __init__(self, name, signal_key):
        super().__init__(name)
        self.signal_key = signal_key
        self._data = {}           # session -> (T, d_s+1) array
        self._class_targets = {}  # session -> (T,) int32
        self.n_classes = 0

    def load_session(self, session, binned_signals):
        if self.signal_key in binned_signals:
            self._data[session] = binned_signals[self.signal_key]
            self._class_targets[session] = self._data[session][:, 0].astype(np.int32)

    def has_session(self, session):
        return session in self._data

    def is_available(self, session, t0, seq_len):
        if session not in self._data:
            return False
        return (self._data[session][t0:t0 + seq_len, -1] > 0.5).any()

    def prepare(self, train_ends):
        self.n_classes = 0
        for session, labels in self._class_targets.items():
            valid = self._data[session][:, -1] > 0.5
            if valid.any():
                self.n_classes = max(self.n_classes, int(labels[valid].max()) + 1)
        print(f"    {self.name}: {self.n_classes} classes")

    def _get_target_batch(self, session, t0s, seq_len, device):
        labels_list = [self._class_targets[session][t0:t0 + seq_len] for t0 in t0s]
        valid_list = [self._data[session][t0:t0 + seq_len, -1] > 0.5 for t0 in t0s]
        labels = torch.from_numpy(np.ascontiguousarray(np.stack(labels_list))).long().to(device)
        valid = torch.from_numpy(np.stack(valid_list)).to(device)  # (B, T)
        return labels.clamp(0, self.n_classes - 1), valid

    def compute_loss(self, h, adapter, session, t0s, seq_len, device, **_):
        labels, valid = self._get_target_batch(session, t0s, seq_len, device)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        head = adapter.task_heads[self.name]
        logits = h @ head.T  # (B, T, n_classes)
        return F.cross_entropy(logits[valid], labels[valid])

    def reset_val_metrics(self):
        self._correct = 0
        self._total = 0

    def update_val_metrics(self, h, adapter, session, t0s, seq_len, device, dt=None, **_):
        labels, valid = self._get_target_batch(session, t0s, seq_len, device)
        if not valid.any():
            return
        head = adapter.task_heads[self.name]
        logits = h @ head.T  # (B, T, n_classes)
        # Rebin to 100ms bins: sum logits, majority vote for labels
        cls_bin = 0.1  # 100ms
        rebin = max(1, round(cls_bin / dt)) if dt else 1
        if rebin > 1:
            B, T, C = logits.shape
            T_trim = (T // rebin) * rebin
            logits = logits[:, :T_trim].reshape(B, T_trim // rebin, rebin, C).sum(dim=2)
            # Majority vote: pick most frequent label in each 100ms bin
            lab_trim = labels[:, :T_trim].reshape(B, T_trim // rebin, rebin)
            labels = torch.mode(lab_trim, dim=2).values
            valid = valid[:, :T_trim].reshape(B, T_trim // rebin, rebin).any(dim=2)
        preds = logits[valid].argmax(dim=1)
        self._correct += (preds == labels[valid]).sum().item()
        self._total += valid.sum().item()

    def get_val_metrics(self):
        if self._total == 0:
            return {}
        return {"cls_acc": self._correct / self._total}


TASK_SPECS = [
    ARTask("ar_visible", target="visible"),
    ARTask("ar_hidden",  target="hidden"),
    RegressionTask("reg/running_speed",       "behavior/running_speed"),
    RegressionTask("reg/pupil_area",          "behavior/pupil_area"),
    RegressionTask("reg/gaze_position",       "behavior/gaze_position"),
    #RegressionTask("reg/flash",               "stimulus/flashes/time_since_last_flash"),
    #RegressionTask("reg/natural_movie_one",   "stimulus/natural_movie_one/movie_position"),
    #RegressionTask("reg/natural_movie_three", "stimulus/natural_movie_three/movie_position"),
    #ClassificationTask("cls/natural_scenes",    "stimulus/natural_scenes/image_index"),
    ClassificationTask("cls/drifting_gratings", "stimulus/drifting_gratings/condition_index"),
    #ClassificationTask("cls/gabors",            "stimulus/gabors/condition_index"),
]


def _get_available_tasks(task_specs, session, t0s, seq_len):
    """Return tasks available for at least one window in the batch."""
    return [spec for spec in task_specs
            if any(spec.is_available(session, t0, seq_len) for t0 in t0s)]


# ---------------------------------------------------------------------------
# Session adapter: maps spike counts <-> model embedding space
#
# The transformer sees ONLY spikes + task embedding. Signals are targets only.
# ---------------------------------------------------------------------------

class LearnedSessionAdapter(nn.Module):
    """Per-session spike embeddings + per-task heads.

    - Per-session w_in (N, d) and w_out (N, d) for spike encoding/decoding
    - Per-session b_out (N,) bias initialized to mean firing rate
    - Per-task embedding (d,) to condition the model on the current task
    """
    def __init__(self, n_units_per_session, task_specs, d, device,
                 mean_rates=None, session_coords=None):
        super().__init__()
        self.d = d
        self.session_coords = [
            torch.tensor(c, dtype=torch.float32, device=device)
            for c in session_coords
        ]
        
        from gradient_normalizer import GradientNormalizer
        self.grad_normalizers = nn.ModuleDict({
            spec.name: GradientNormalizer(normalizer_shape=[1], scale=1.0)
            for spec in task_specs
        })
        self.task_embeds = nn.ParameterDict({
            spec.name: nn.Parameter(torch.randn(d, device=device) * 0.02)
            for spec in task_specs
        })
        # self._unit_in = nn.ParameterList([
        #     nn.Parameter(torch.randn(n, d, device=device) * (1.0 / n**0.5))
        #     for n in n_units_per_session
        # ])
        self.coord_mlp_in = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, d),
        )
        # self._unit_out = nn.ParameterList([
        #     nn.Parameter(torch.zeros(n, d, device=device))
        #     for n in n_units_per_session
        # ])
        self.coord_mlp_out = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, d),
        )
        # b_out initialized so that relu(0 + b_out) = mean firing rate per neuron
        self._b_out = nn.ParameterList([
            nn.Parameter(torch.from_numpy(mean_rates[i]).float().to(device))
            for i in range(len(n_units_per_session))
        ])
        # Per-task prediction heads
        task_heads = {}
        for spec in task_specs:
            if isinstance(spec, RegressionTask):
                task_heads[spec.name] = nn.Parameter(
                    torch.randn(spec.d_out, d, device=device) * (1.0 / d**0.5))
            elif isinstance(spec, ClassificationTask):
                task_heads[spec.name] = nn.Parameter(
                    torch.randn(spec.n_classes, d, device=device) * (1.0 / d**0.5))
        self.task_heads = nn.ParameterDict(task_heads)

    def w_in(self, session, visible_idx=None):
        # w = self._unit_in[session]
        # return w[visible_idx] if visible_idx is not None else w
        coords = self.session_coords[session]
        if visible_idx is not None:
            coords = coords[visible_idx]
        return self.coord_mlp_in(coords)

    def w_out(self, session):
        # return self._unit_out[session]
        return self.coord_mlp_out(self.session_coords[session])

    def b_out(self, session):
        return self._b_out[session]



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wrapper(name, cache_dir, data_root):
    if name == "allen":
        from spike_data.allen import AllenNeuropixelsWrapper
        return AllenNeuropixelsWrapper(
            cache_dir=cache_dir,
            data_root=f"{data_root}/allen_neuropixels/ecephys_cache_dir")
    raise ValueError(f"Unknown dataset: {name}")


def load_raster(wrapper, session, dt):
    """Load a single session's raster (uses wrapper's .npy cache)."""
    return wrapper.get_raster_plot(session, t_start=0.0, t_end=1e9, delta_t=dt)


def load_rasters(wrapper, session_indices, dt):
    """Load rasters for a list of sessions. Returns dict: session -> raster."""
    rasters = {}
    for session in session_indices:
        cached = wrapper._raster_cache_path(session, dt).exists()
        print(f"  Session {session}: raster {'from cache' if cached else 'computing...'}", end="", flush=True)
        t0 = time.time()
        rasters[session] = load_raster(wrapper, session, dt)
        print(f"  shape={rasters[session].shape}  ({time.time() - t0:.1f}s)")
    return rasters


def load_binned_signals(wrapper, session_indices, dt, rasters):
    """Load binned signals. Returns dict: session -> binned_signals."""
    all_binned = {}
    for session in session_indices:
        T = rasters[session].shape[1]
        t0 = time.time()
        binned, _ = wrapper.get_binned_signals(session, delta_t=dt, n_bins=T)
        print(f"  Session {session}: {len(binned)} signals ({time.time() - t0:.1f}s)")
        all_binned[session] = binned
    return all_binned




def get_session_metadata(wrapper, session, dt):
    """Get unit count and val_start_bin for a single session (must be cached)."""
    raster = load_raster(wrapper, session, dt)
    n_units = raster.shape[0]
    T = raster.shape[1]
    signals = wrapper.get_signals(session, 0.0, 1e9)
    onset_key = "stimulus/drifting_gratings/onset"
    if onset_key in signals and len(signals[onset_key][0]) > 0:
        last_onset = float(signals[onset_key][0][-1])
        val_start = max(0, int((last_onset - VAL_STIM_DURATION_S) / dt))
    else:
        val_start = T
    del raster
    return n_units, val_start


VAL_STIM_DURATION_S = 2 * 60



def make_windows_by_session(rasters, seq_len, val_start_bins, mode="train",
                            n_val_windows=100):
    """Return dict: session -> list of valid bin_start positions.

    For mode="val", pick n_val_windows evenly spaced windows per session
    (deterministic for reproducibility). For mode="train", return all windows."""
    windows = {}
    for session, raster in rasters.items():
        T = raster.shape[1]
        if mode == "train":
            t_start, t_end = 0, val_start_bins[session]
        else:
            t_start, t_end = val_start_bins[session], T
        t0s = list(range(t_start, t_end - seq_len, seq_len))
        if mode == "val" and len(t0s) > n_val_windows:
            step = len(t0s) / n_val_windows
            t0s = [t0s[int(i * step)] for i in range(n_val_windows)]
        if t0s:
            windows[session] = t0s
    return windows


def make_batched_steps(windows_by_session, batch_size):
    """Create a shuffled list of (session, [t0_1, ..., t0_B]) batches.

    Each batch samples B distant windows from the same session.
    If a session has fewer windows than batch_size, use all of them."""
    steps = []
    for session, t0s in windows_by_session.items():
        random.shuffle(t0s)
        for i in range(0, len(t0s), batch_size):
            batch_t0s = t0s[i:i + batch_size]
            steps.append((session, batch_t0s))
    random.shuffle(steps)
    return steps


# ---------------------------------------------------------------------------
# Unit masking
# ---------------------------------------------------------------------------

def _mask_two_bands(N, mask_frac):
    n_mask = max(1, int(mask_frac * N))
    n1 = random.randint(1, max(1, n_mask - 1))
    n2 = n_mask - n1
    start1 = random.randint(0, N - 1)
    start2 = random.randint(0, N - 1)
    masked = set()
    for i in range(n1):
        masked.add((start1 + i) % N)
    for i in range(n2):
        masked.add((start2 + i) % N)
    hidden_idx = sorted(masked)
    visible_idx = sorted(set(range(N)) - masked)
    if not visible_idx:
        visible_idx = [0]
        hidden_idx = sorted(set(range(N)) - {0})
    return visible_idx, hidden_idx


def _sample_mask(N, mask_frac):
    n_mask = max(1, int(mask_frac * N))
    if random.random() < 0.5:
        return _mask_two_bands(N, mask_frac)
    masked = set(random.sample(range(N), n_mask))
    hidden_idx = sorted(masked)
    visible_idx = sorted(set(range(N)) - masked)
    if not visible_idx:
        visible_idx = [0]
        hidden_idx = sorted(set(range(N)) - {0})
    return visible_idx, hidden_idx


# ---------------------------------------------------------------------------
# SOS tokens and document-aware attention
# ---------------------------------------------------------------------------

def _make_sos_and_block_mask(seq_len, batch_size, dt, device):
    sos = torch.zeros(seq_len, device=device, dtype=torch.bool)
    sos[0] = True
    sos[1:] = torch.rand(seq_len - 1, device=device) < (dt / 2.0)
    doc_ids = sos.long().cumsum(0)
    arange = torch.arange(seq_len, device=device, dtype=torch.float)
    last_sos = (sos.float() * arange).cummax(0).values
    positions = arange - last_sos

    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])
    # Mask is identical across batch elements (doc_ids doesn't depend on b),
    # so use B=None to avoid redundant computation and let flex_attention broadcast.
    block_mask = create_block_mask(mask_fn, B=None, H=None,
                                   Q_LEN=seq_len, KV_LEN=seq_len, device=device)
    return sos, positions, block_mask


def _precompute_sos_pool(n_masks, seq_len, batch_size, dt, device):
    print(f"  Pre-computing {n_masks} SOS block masks (seq_len={seq_len}, B={batch_size})...", end="", flush=True)
    t0 = time.time()
    pool = [_make_sos_and_block_mask(seq_len, batch_size, dt, device) for _ in range(n_masks)]
    print(f" done ({time.time() - t0:.1f}s)")
    return pool


# ---------------------------------------------------------------------------
# Autoregressive sampling (for validation raster plot)
# ---------------------------------------------------------------------------

def _sample_autoregressive(model, adapter, rasters, session, t0,
                           n_context, n_generate, device, block_mask):
    raster = rasters[session]
    N = raster.shape[0]
    n_total = n_context + n_generate
    visible_idx, hidden_idx = _sample_mask(N, 0.5)

    z_real = (raster[:, t0:t0 + n_total] > 0).astype(np.float32)
    z_ctx = torch.from_numpy(
        np.ascontiguousarray(z_real[:, :n_context].T)).to(device)

    x_seq = z_ctx[:, visible_idx] @ adapter.w_in(session, visible_idx) \
        + adapter.task_embeds["ar_visible"]

    # Inject SOS at position 0 (consistent with training)
    sos_embed = (model.module.sos_embed if hasattr(model, 'module')
                 else model._orig_mod.sos_embed).float()
    x_seq[0] = sos_embed

    # Predict z for the clamped context segment (next-step prediction)
    bm_len = block_mask.shape[-1]
    if n_context <= bm_len:
        x_padded = F.pad(x_seq, (0, 0, 0, bm_len - n_context))
        h_ctx = model(x_padded.bfloat16().unsqueeze(0), block_mask).squeeze(0)
        h_ctx = h_ctx[:n_context].float()
    else:
        x_padded = x_seq[-bm_len:]
        h_ctx = model(x_padded.bfloat16().unsqueeze(0), block_mask).squeeze(0).float()
    z_pred_ctx = stochastic_binary_spike(h_ctx @ adapter.w_out(session).T + adapter.b_out(session)).cpu().numpy().T  # (N, T_ctx)

    generated_z = []
    for _ in range(n_generate):
        real_len = x_seq.shape[0]
        if real_len < bm_len:
            x_padded = F.pad(x_seq, (0, 0, 0, bm_len - real_len))
            h_idx = real_len - 1
        else:
            x_padded = x_seq[-bm_len:]
            h_idx = bm_len - 1
        h = model(x_padded.bfloat16().unsqueeze(0), block_mask).squeeze(0)
        h_last = h[h_idx].float()

        z_pred = stochastic_binary_spike((h_last.unsqueeze(0) @ adapter.w_out(session).T + adapter.b_out(session)).squeeze(0))
        generated_z.append(z_pred.cpu().numpy())

        x_next = z_pred[visible_idx].unsqueeze(0) @ adapter.w_in(session, visible_idx) \
            + adapter.task_embeds["ar_visible"]
        x_seq = torch.cat([x_seq, x_next], dim=0)

    generated_z = np.stack(generated_z).T
    z_gen = np.concatenate([z_real[:, :n_context], generated_z], axis=1)
    return dict(visible_idx=visible_idx, hidden_idx=hidden_idx,
                z_real=z_real, z_gen=z_gen, z_pred_ctx=z_pred_ctx)


def _plot_raster_comparison(fig_raster, sample, n_context, dt, n_show=25):
    fig_raster.clear()
    axes = fig_raster.subplots(3, 1, sharex=True)
    ax_vis, ax_hid_gen, ax_hid_real = axes

    vis = sample["visible_idx"][:n_show]
    hid = sample["hidden_idx"][:n_show]
    z_real, z_gen = sample["z_real"], sample["z_gen"]
    T = z_real.shape[1]
    t_axis = np.arange(T) * dt * 1000
    context_ms = n_context * dt * 1000

    def _draw(ax, unit_ids, z, label):
        for i, uid in enumerate(unit_ids):
            spk = t_axis[z[uid] > 0]
            ax.scatter(spk, np.full_like(spk, i), s=4, c="k", marker="|", lw=0.5)
        ax.axvline(context_ms, color="red", lw=1.5, ls="--")
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(-0.5, len(unit_ids) - 0.5)
        ax.set_yticks([])

    _draw(ax_vis, vis, z_gen, "Visible\n(generated)")
    _draw(ax_hid_gen, hid, z_gen, "Hidden\n(generated)")
    _draw(ax_hid_real, hid, z_real, "Hidden\n(ground truth)")

    ax_hid_real.set_xlabel("Time (ms)")
    fig_raster.suptitle("Context \u2192 autoregressive generation", fontsize=10)
    fig_raster.tight_layout()
    fig_raster.canvas.draw(); fig_raster.canvas.flush_events()
    fig_raster.savefig("val_raster.png", dpi=100)


def _update_plot(fig_ax, h):
    fig, axes = fig_ax
    for ax in axes.flat:
        ax.clear()
    ax_l1, ax_reg, ax_cls = axes[0]
    steps = h["train_step"]
    for k, vals in h.get("train_ema", {}).items():
        s = steps[:len(vals)]
        if k.startswith("ar_"):
            ax_l1.plot(s, vals, lw=0.8, label=k.replace("ar_", ""))
        elif k.startswith("reg/"):
            ax_reg.plot(s, vals, lw=0.8, label=k.replace("reg/", ""))
        elif k.startswith("cls/"):
            ax_cls.plot(s, vals, lw=0.8, label=k.replace("cls/", ""))
    ax_l1.set_ylabel("Spike loss"); ax_l1.set_xlabel("Step"); ax_l1.legend(fontsize=7)
    ax_reg.set_ylabel("Regression MSE"); ax_reg.set_xlabel("Step"); ax_reg.legend(fontsize=7)
    ax_cls.set_ylabel("Class CE"); ax_cls.set_xlabel("Step")
    ep = h["epoch"]
    ax_f1, ax_r2, ax_acc = axes[1]
    if ep:
        ax_f1.plot(ep, h["val_spike_f1"], "b-o", ms=3)
        ax_f1.set_ylabel("Spike F1"); ax_f1.set_xlabel("Epoch"); ax_f1.set_ylim(0, 1)
        for k, vals in h["val_r2"].items():
            ax_r2.plot(ep[:len(vals)], vals, "-o", ms=3, label=k)
        ax_r2.set_ylabel("Regression R\u00b2"); ax_r2.set_xlabel("Epoch"); ax_r2.legend(fontsize=6)
        ax_acc.plot(ep, h["val_cls_acc"], "m-o", ms=3)
        ax_acc.set_ylabel("Class accuracy"); ax_acc.set_xlabel("Epoch"); ax_acc.set_ylim(0, 1)
    fig.tight_layout()
    fig.canvas.draw(); fig.canvas.flush_events()
    fig.savefig("training_curves.png", dpi=100)


def _run_validation(model, adapter, rasters, val_start_bins, sos_pool,
                    causal_block_mask, seq_len, B, dt, device,
                    fig_raster=None):
    """Run validation metrics + autoregressive sample. Returns (metrics, n_val)."""
    model.eval()
    adapter.eval()
    # Fixed seed for reproducible validation (same windows, masks, Bernoulli samples)
    rng_state = random.getstate()
    torch_rng_state = torch.random.get_rng_state()
    random.seed(0)
    torch.manual_seed(0)
    sos_embed_val = (model.module.sos_embed if hasattr(model, 'module')
                     else model._orig_mod.sos_embed).float()
    val_by_sess = make_windows_by_session(rasters, seq_len, val_start_bins, mode="val")
    val_batched = make_batched_steps(val_by_sess, B)

    for spec in TASK_SPECS:
        spec.reset_val_metrics()
    n_val = 0

    with torch.no_grad():
        for session, batch_t0s in val_batched:
            cur_B = len(batch_t0s)
            z_np = np.stack([(rasters[session][:, t0:t0 + seq_len] > 0).astype(np.float32).T
                             for t0 in batch_t0s])
            z = torch.from_numpy(np.ascontiguousarray(z_np)).to(device)
            N = z.shape[2]
            visible_idx, hidden_idx = _mask_two_bands(N, 0.3)
            sos_mask, positions, seg_block_mask = sos_pool[0]
            n_val += cur_B

            for spec in TASK_SPECS:
                if not spec.has_session(session):
                    continue
                task_embed = adapter.task_embeds[spec.name]
                x = z[:, :, visible_idx] @ adapter.w_in(session, visible_idx) + task_embed
                x[:, sos_mask] = sos_embed_val + task_embed
                if cur_B < B:
                    x_padded = F.pad(x, (0, 0, 0, 0, 0, B - cur_B))
                else:
                    x_padded = x
                h = model(x_padded.bfloat16(), seg_block_mask, positions)[:cur_B].float()
                spec.update_val_metrics(
                    h, adapter, session, batch_t0s, seq_len, device,
                    z=z, visible_idx=visible_idx, hidden_idx=hidden_idx, dt=dt)

    all_metrics = {}
    for spec in TASK_SPECS:
        all_metrics.update(spec.get_val_metrics())

    # Autoregressive raster sample
    ar_stats = None
    if val_batched:
        n_gen = int(1.0 / dt)
        n_ctx = int(1.0 / dt)
        session_s, t0s_s = val_batched[0]
        t0_s = t0s_s[0]
        with torch.no_grad():
            sample = _sample_autoregressive(
                model, adapter, rasters, session_s, t0_s, n_ctx, n_gen,
                device, causal_block_mask)
        if fig_raster is not None:
            _plot_raster_comparison(fig_raster, sample, n_ctx, dt)
        ar_stats = {
            "ctx_pred": (sample["z_pred_ctx"] > 0).mean().item(),
            "ctx_real": sample["z_real"][:, :n_ctx].mean().item(),
            "gen_pred": sample["z_gen"][:, n_ctx:].mean().item(),
            "gen_real": sample["z_real"][:, n_ctx:].mean().item(),
        }

    # Restore RNG state so training randomness is unaffected
    random.setstate(rng_state)
    torch.random.set_rng_state(torch_rng_state)

    return all_metrics, n_val, ar_stats


def _print_val(all_metrics, n_val, ar_stats, prefix="  "):
    """Print validation metrics and activity rates."""
    val_f1 = all_metrics.get("spike_f1", 0.0)
    val_r2 = {k: v for k, v in all_metrics.items() if k.startswith("r2/")}
    val_cls_acc = all_metrics.get("cls_acc", 0.0)
    r2_parts = [f"{k}={v:.3f}" for k, v in sorted(val_r2.items())]
    print(f"{prefix}Val: F1={val_f1:.3f}  "
          + (f"R\u00b2: {', '.join(r2_parts)}  " if r2_parts else "")
          + (f"Acc: {val_cls_acc:.3f}  " if "cls_acc" in all_metrics else "")
          + f"({n_val} windows)")
    if ar_stats:
        print(f"{prefix}Activity rate: clamped pred={ar_stats['ctx_pred']:.4f} "
              f"real={ar_stats['ctx_real']:.4f}"
              f"  |  generated={ar_stats['gen_pred']:.4f} "
              f"real={ar_stats['gen_real']:.4f}")
    return val_f1, val_r2, val_cls_acc


def _eval_only(args):
    """Load checkpoint from a previous run, run validation, and exit."""
    run_dir = Path(args.eval_only)
    ckpt_path = run_dir / "checkpoint.pt"
    assert ckpt_path.exists(), f"No checkpoint found at {ckpt_path}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    model_name = saved_args["model"]
    dt = args.dt
    print(f"Eval-only: {run_dir}  model={model_name}  dt={dt}")

    # Load data
    datasets = [s.strip() for s in args.datasets.split(",")]
    wrappers, all_session_indices, n_units_all, val_start_bins = [], [], [], {}
    for ds_name in datasets:
        wrapper = load_wrapper(ds_name, cache_dir=args.cache_dir, data_root=args.data_root)
        n_total = wrapper.get_number_of_sessions()
        offset = len(all_session_indices)
        for local_s in range(min(n_total, args.n_max_sessions)):
            global_s = offset + local_s
            n_u, vs = get_session_metadata(wrapper, local_s, dt)
            all_session_indices.append((len(wrappers), local_s))
            n_units_all.append(n_u)
            val_start_bins[global_s] = vs
        wrappers.append(wrapper)

    rasters = {}
    for global_s in range(len(all_session_indices)):
        w_idx, local_s = all_session_indices[global_s]
        rasters[global_s] = load_raster(wrappers[w_idx], local_s, dt)
        T = rasters[global_s].shape[1]
        binned, _ = wrappers[w_idx].get_binned_signals(local_s, delta_t=dt, n_bins=T)
        for spec in TASK_SPECS:
            spec.load_session(global_s, binned)
    for spec in TASK_SPECS:
        spec.prepare({s: val_start_bins[s] for s in rasters})

    # Rebuild model + adapter, load weights
    config = CONFIGS[model_name]
    model = SpikeGPT(config).cuda().bfloat16()
    model = torch.compile(model)
    B = args.batch_size

    mean_rates = []
    for s in range(len(n_units_all)):
        if s in rasters:
            p = (rasters[s][:, :val_start_bins[s]] > 0).mean(axis=1).clip(1e-6, 1 - 1e-6)
            mean_rates.append(np.log(p / (1 - p)))
        else:
            mean_rates.append(np.zeros(n_units_all[s]))
    adapter = LearnedSessionAdapter(n_units_all, TASK_SPECS, config.n_embd, device,
                                    mean_rates=mean_rates)
    adapter.to(device)
    model.load_state_dict(ckpt["model"])
    adapter.load_state_dict(ckpt["adapter"])

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    causal_block_mask = create_block_mask(
        causal_mask, B=1, H=None, Q_LEN=args.seq_len, KV_LEN=args.seq_len,
        device="cuda", _compile=True)
    sos_pool = _precompute_sos_pool(args.n_sos_masks, args.seq_len, B, dt, device)

    all_metrics, n_val, ar_stats = _run_validation(
        model, adapter, rasters, val_start_bins, sos_pool,
        causal_block_mask, args.seq_len, B, dt, device)
    _print_val(all_metrics, n_val, ar_stats, prefix="")

    with open(run_dir / "eval_results.json", "w") as f:
        json.dump({"eval_args": vars(args), "metrics": all_metrics}, f, indent=2)
    print(f"Saved to {run_dir / 'eval_results.json'}")


class _Tee:
    def __init__(self, log_path):
        self._file = open(log_path, "w")
        self._stdout = sys.stdout
    def write(self, s):
        self._stdout.write(s)
        self._file.write(s)
    def flush(self):
        self._stdout.flush()
        self._file.flush()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    if args.eval_only:
        return _eval_only(args)

    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(run_dir / "train.log")
    print(f"Run directory: {run_dir}")
    print(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    datasets = [s.strip() for s in args.datasets.split(",")]
    wrappers = []
    all_session_indices = []  # global session index -> (wrapper_idx, local_session_idx)
    n_units_all = []
    val_start_bins = {}  # global session -> val start bin

    for ds_name in datasets:
        print(f"Loading dataset: {ds_name}")
        wrapper = load_wrapper(ds_name, cache_dir=args.cache_dir, data_root=args.data_root)
        n_total = wrapper.get_number_of_sessions()
        if args.use_all_sessions:
            # Discover sessions already cached + first n_max_sessions (download if needed)
            cached = [s for s in range(n_total)
                      if wrapper._raster_cache_path(s, args.dt).exists()]
            must_have = list(range(min(n_total, args.n_max_sessions)))
            discover_list = sorted(set(cached) | set(must_have))
            print(f"  {len(cached)} sessions cached, "
                  f"discovering {len(discover_list)} total "
                  f"(rotating {args.n_max_sessions} at a time)")
        else:
            discover_list = list(range(min(n_total, args.n_max_sessions)))
            print(f"  Using {len(discover_list)} sessions")

        # Get metadata for discovered sessions
        offset = len(all_session_indices)
        for local_s in discover_list:
            global_s = offset + local_s
            n_u, vs = get_session_metadata(wrapper, local_s, args.dt)
            all_session_indices.append((len(wrappers), local_s))
            n_units_all.append(n_u)
            val_start_bins[global_s] = vs
        wrappers.append(wrapper)

    n_total_sessions = len(all_session_indices)
    print(f"Total sessions discovered: {n_total_sessions}")
    print(f"Units per session: {n_units_all}")

    # Load initial subset of sessions
    if args.use_all_sessions:
        active_sessions = sorted(random.sample(range(n_total_sessions),
                                               min(args.n_max_sessions, n_total_sessions)))
    else:
        active_sessions = list(range(n_total_sessions))

    def _load_active_rasters(session_indices):
        """Load rasters for the given global session indices."""
        rasters = {}
        for global_s in session_indices:
            w_idx, local_s = all_session_indices[global_s]
            rasters[global_s] = load_raster(wrappers[w_idx], local_s, args.dt)
        return rasters

    # Pre-load signals for ALL discovered sessions (cached, so fast).
    # This ensures normalization stats and n_classes are global.
    print("Pre-loading signals for all sessions...")
    for global_s in range(n_total_sessions):
        w_idx, local_s = all_session_indices[global_s]
        raster_shape = load_raster(wrappers[w_idx], local_s, args.dt).shape
        T = raster_shape[1]
        binned, _ = wrappers[w_idx].get_binned_signals(local_s, delta_t=args.dt, n_bins=T)
        for spec in TASK_SPECS:
            spec.load_session(global_s, binned)
        del raster_shape

    # Prepare tasks using ALL sessions' training data
    train_ends = {s: val_start_bins[s] for s in range(n_total_sessions)}
    print("  Preparing tasks:")
    for spec in TASK_SPECS:
        spec.prepare(train_ends)

    print(f"Loading initial sessions: {active_sessions}")
    rasters = _load_active_rasters(active_sessions)

    # Print task availability
    for spec in TASK_SPECS:
        if isinstance(spec, ARTask):
            print(f"  Task {spec.name}: all sessions")
        else:
            hits = [s for s in active_sessions if spec.has_session(s)]
            print(f"  Task {spec.name}: {len(hits)} sessions {hits}")

    # --- Model ---
    config = CONFIGS[args.model]
    d = config.n_embd
    model = SpikeGPT(config).cuda().bfloat16()
    model = torch.compile(model)

    B = args.batch_size

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    causal_block_mask = create_block_mask(
        causal_mask, B=1, H=None, Q_LEN=args.seq_len, KV_LEN=args.seq_len,
        device="cuda", _compile=True)
    sos_pool = _precompute_sos_pool(args.n_sos_masks, args.seq_len, B, args.dt, device)

    # Log-odds of spike probability per neuron (training data only) — used to init b_out
    # Computed from currently loaded sessions; sessions loaded later get zeros
    mean_rates = []
    for s in range(n_total_sessions):
        if s in rasters:
            train_end = val_start_bins[s]
            p = (rasters[s][:, :train_end] > 0).mean(axis=1).clip(1e-6, 1 - 1e-6)
            mean_rates.append(np.log(p / (1 - p)))
        else:
            mean_rates.append(np.zeros(n_units_all[s]))
    adapter = LearnedSessionAdapter(n_units_all, TASK_SPECS, d, device,
                                    mean_rates=mean_rates)
    adapter.to(device)
    fast_params = [p for p in adapter.parameters() if p.ndim < 2] \
        + [p for p in model.parameters() if p.ndim < 2]
    matrix_params = [p for p in adapter.parameters() if p.ndim >= 2] \
        + [p for p in model.parameters() if p.ndim >= 2]
    all_params = fast_params + matrix_params
    assert len({id(p) for p in all_params}) == len(all_params)
    opt_fast = torch.optim.Adam(fast_params, lr=args.lr * 10)
    opt_matrix = torch.optim.AdamW(matrix_params, lr=args.lr, weight_decay=0.1)

    n_params = sum(p.numel() for p in all_params) / 1e6
    print(f"Model: {n_params:.2f}M parameters ({len(fast_params)} fast, {len(matrix_params)} matrix)")

    warmup = lambda step: min(1.0, step / args.warmup_steps) if args.warmup_steps > 0 else 1.0
    sched_fast = torch.optim.lr_scheduler.LambdaLR(opt_fast, warmup)
    sched_matrix = torch.optim.lr_scheduler.LambdaLR(opt_matrix, warmup)

    # --- Live plots ---
    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig_raster = plt.figure(figsize=(10, 5))
    hist = dict(train_step=[], train_ema={},
                epoch=[], val_spike_f1=[], val_r2={}, val_cls_acc=[])

    # --- Pre-training sanity check ---
    _, _, ar_stats = _run_validation(
        model, adapter, rasters, val_start_bins, sos_pool,
        causal_block_mask, args.seq_len, B, args.dt, device,
        fig_raster=fig_raster)
    if ar_stats:
        print(f"Pre-train activity rate: clamped pred={ar_stats['ctx_pred']:.4f} "
              f"real={ar_stats['ctx_real']:.4f}"
              f"  |  generated={ar_stats['gen_pred']:.4f} "
              f"real={ar_stats['gen_real']:.4f}")

    # --- Training loop ---
    model.train()
    start_time = time.time()
    total_eval_time = 0.0
    ema = {}  # task_name -> EMA loss value
    global_step = 0

    for epoch in range(1, args.n_epochs + 1):
        # Rotate sessions if using all sessions
        if args.use_all_sessions and epoch > 1:
            active_sessions = sorted(random.sample(range(n_total_sessions),
                                                   min(args.n_max_sessions, n_total_sessions)))
            print(f"  Rotating sessions: {active_sessions}")
            rasters = _load_active_rasters(active_sessions)

        win_by_sess = make_windows_by_session(rasters, args.seq_len, val_start_bins, mode="train")
        batched_steps = make_batched_steps(win_by_sess, B)
        n_steps = len(batched_steps)
        log_every = min(args.log_every, max(1, n_steps // 10))
        print(f"\nEpoch {epoch}/{args.n_epochs}  ({n_steps} batched steps, B={B})")

        for step_i, (session, batch_t0s) in enumerate(batched_steps, 1):
            global_step += 1
            cur_B = len(batch_t0s)
            opt_fast.zero_grad()
            opt_matrix.zero_grad()

            # Pick tasks (check availability across all windows in batch)
            available = _get_available_tasks(TASK_SPECS, session, batch_t0s, args.seq_len)
            n_tasks = random.randint(1, len(available))
            step_tasks = random.sample(available, n_tasks)

            # Spike input — binarized, batched: (B, T, N)
            z_np = np.stack([(rasters[session][:, t0:t0 + args.seq_len] > 0).astype(np.float32).T
                             for t0 in batch_t0s])
            z = torch.from_numpy(np.ascontiguousarray(z_np)).to(device)  # (B, T, N)
            N = z.shape[2]
            mask_frac = random.uniform(args.mask_range[0], args.mask_range[1])
            visible_idx, hidden_idx = _sample_mask(N, mask_frac)

            # SOS + document-aware attention
            sos_mask, positions, seg_block_mask = sos_pool[global_step % len(sos_pool)]
            sos_embed = (model.module.sos_embed if hasattr(model, 'module')
                         else model._orig_mod.sos_embed)

            # Task embedding: single task or mean of selected tasks
            if n_tasks == 1:
                task_embed = adapter.task_embeds[step_tasks[0].name]
            else:
                task_embed = torch.stack([adapter.task_embeds[t.name] for t in step_tasks]).mean(0)

            n_hops = 1
            if n_tasks == 1 and isinstance(step_tasks[0], ARTask) and args.n_hops > 1:
                n_hops = 1 if np.random.rand() > 1/args.n_hops else args.n_hops

            z_ar = z
            hop_losses = []
            for k in range(n_hops):
                # Batched encode: (B, T, N_vis) @ (N_vis, d) -> (B, T, d)
                x = z_ar[:, :, visible_idx] @ adapter.w_in(session, visible_idx) + task_embed

                x[:, sos_mask] = sos_embed.float() + task_embed

                # Pad batch if smaller than B (block_mask expects fixed B)
                if cur_B < B:
                    x_padded = F.pad(x, (0, 0, 0, 0, 0, B - cur_B))
                else:
                    x_padded = x

                if n_hops > 1:
                    h_full = torch.utils.checkpoint.checkpoint(
                        model, x_padded.bfloat16(), seg_block_mask, positions,
                        use_reentrant=False)[:cur_B].float()
                else:
                    h_full = model(x_padded.bfloat16(), seg_block_mask, positions)[:cur_B].float()

                # Compute loss for each selected task
                task_losses = {}
                for task in step_tasks:
                    h_t = adapter.grad_normalizers[task.name](h_full) if not args.no_grad_norm else h_full
                    task_losses[task.name] = task.compute_loss(
                        h_t, adapter, session, batch_t0s, args.seq_len, device,
                        z=z, visible_idx=visible_idx, hidden_idx=hidden_idx)
                hop_losses.append(sum(task_losses.values()) / len(task_losses))

                if k < n_hops - 1:
                    z_ar = stochastic_binary_spike(h_full @ adapter.w_out(session).T + adapter.b_out(session))
                    z_ar = torch.cat([z[:, 0:1], z_ar[:, :-1]], 1)

            step_loss = sum(hop_losses) / len(hop_losses)
            step_loss.backward()
            # Track EMA per task
            for task in step_tasks:
                v = task_losses[task.name].item()
                ema[task.name] = v if task.name not in ema else 0.99 * ema[task.name] + 0.01 * v

            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt_fast.step()
            opt_matrix.step()
            sched_fast.step()
            sched_matrix.step()

            if step_i % log_every == 0 or step_i == n_steps:
                short = {"ar_visible": "vis", "ar_hidden": "hid",
                         "reg/running_speed": "run", "reg/pupil_area": "pup",
                         "reg/gaze_position": "gaze", "cls/drifting_gratings": "cls"}
                parts = [f"{short.get(k,k)}={v:.4f}" for k, v in sorted(ema.items())]
                print(f"  [{step_i}/{n_steps}]  {'  '.join(parts)}  time={time.time()-start_time:.1f}s")
                hist["train_step"].append(global_step)
                hist["train_ema"] = hist.get("train_ema", {})
                for k, v in ema.items():
                    hist["train_ema"].setdefault(k, []).append(v)
                _update_plot((fig, axes), hist)

        # ----- Validation -----
        t_val_start = time.time()
        all_metrics, n_val, ar_stats = _run_validation(
            model, adapter, rasters, val_start_bins, sos_pool,
            causal_block_mask, args.seq_len, B, args.dt, device,
            fig_raster=fig_raster)
        t_val = time.time() - t_val_start
        total_eval_time += t_val
        t_train = time.time() - start_time - total_eval_time
        val_f1, val_r2, val_cls_acc = _print_val(all_metrics, n_val, ar_stats)
        print(f"  Time: train={t_train:.1f}s  eval={total_eval_time:.1f}s  ratio={total_eval_time/max(t_train,1):.2f}")

        hist["epoch"].append(epoch)
        hist["val_spike_f1"].append(val_f1)
        for k, v in val_r2.items():
            hist["val_r2"].setdefault(k, []).append(v)
        hist["val_cls_acc"].append(val_cls_acc)
        _update_plot((fig, axes), hist)

        model.train()
        adapter.train()

    # Collect final validation metrics
    final_metrics = {}
    for spec in TASK_SPECS:
        final_metrics.update(spec.get_val_metrics())

    # Save
    torch.save({"model": model.state_dict(), "adapter": adapter.state_dict(),
                "args": vars(args)}, run_dir / "checkpoint.pt")
    fig.savefig(run_dir / "training_curves.png", dpi=150)
    fig_raster.savefig(run_dir / "val_raster.png", dpi=150)
    results = {"args": vars(args), "history": hist, "final_val": final_metrics}
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {run_dir}")
    plt.ioff()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike GPT")
    parser.add_argument("--model", type=str, default="tiny", choices=list(CONFIGS.keys()))
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--datasets", type=str, default="allen")
    parser.add_argument("--n_max_sessions", type=int, default=20)
    parser.add_argument("--use_all_sessions", action="store_true",
                        help="Rotate through all sessions, loading n_max_sessions at a time")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask_range", type=str, default="0.1,0.5")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_tasks_per_step", type=int, default=None,
                        help="(deprecated, ignored) Tasks are randomly sampled 1..all per step")
    parser.add_argument("--no_grad_norm", action="store_true", default=True,
                        help="Disable gradient normalizer on task heads")
    parser.add_argument("--n_sos_masks", type=int, default=128)
    parser.add_argument("--n_hops", type=int, default=10)
    parser.add_argument("--eval_only", type=str, default=None, metavar="RUN_DIR",
                        help="Load checkpoint from RUN_DIR, run validation, and exit")
    parser.add_argument("--cache_dir", type=str, default="/scratch/neuro-ai/cache")
    parser.add_argument("--data_root", type=str, default="/share/datasets/neuro-ai")
    args = parser.parse_args()
    args.mask_range = tuple(float(x) for x in args.mask_range.split(","))
    if args.seq_len is None:
        args.seq_len = 16 * CONFIGS[args.model].window_size
    train(args)
