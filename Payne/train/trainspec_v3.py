"""trainspec_v3.py

Three-phase serialized trainer for SpectralMLP_v3.

Training philosophy
-------------------
Stellar spectra are dominated by their intrinsic stellar parameters; dust
extinction is a secondary (and physically separable) modifier; and any
residual non-linearity is tertiary.  Training all three heads simultaneously
from random initialisation leads to gradient interference: the f0 head
dominates the loss early, pulling the extinction and residual lanes into
compensating for stellar error rather than learning their intended physics.

The three-phase strategy avoids this:

  Phase 1 — intrinsic only
      Only f0 is trained.  Extinction and residual lanes are disabled.
      The stellar head learns a clean mapping from physical params to
      log10 flux, uncontaminated by dust.

  Phase 2 — add extinction
      f0 weights are loaded from phase 1.  f0 is frozen for the first
      `warm_freeze_epochs` epochs (default 50) so the khat head can find
      a sensible initialisation before f0 is allowed to co-adapt.
      After warm-in, both f0 and khat are trained jointly.

  Phase 3 — add residual
      f0 + khat weights loaded from phase 2.  Both are frozen for the
      warm-in period, then all three heads train together.  The residual
      gate (res_gate) is initialised at a low value (sigmoid(-2) ≈ 0.12)
      so the residual lane starts silent and grows only as needed.

Each phase has its own cosine LR schedule with a linear warm-up, so newly
activated heads always start with adequate learning rate regardless of how
far through training the previous phase ended.

Usage
-----
    from trainspec_v3 import TrainSpec

    trainer = TrainSpec(
        modpath      = '/data/grid/',
        output       = 'run1.h5',
        wave_range   = (4000.0, 9000.0),
        R            = 100.0,
        n_epochs_p1  = 5000,
        n_epochs_p2  = 3000,
        n_epochs_p3  = 2000,
    )
    trainer.run(basis_K=100)
"""
from __future__ import annotations

import math
import os
import time
from datetime import datetime
from functools import partial
from typing import Optional
import gc

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
mp.set_start_method('fork', force=True)

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step")

# local imports (adjust to your package layout)
from ..utils import readKorg
from ..utils.readKorg import XYFromFlat, _worker_init_fn
from ..utils.io_h5 import (
    save_state_dict_to_h5,
    load_state_dict_from_h5,
    save_labels_norms_to_h5,
    save_meta_to_h5,
)
from .NNmodels_new_v3 import SpectralMLP_v3

# -----------------------------------------------------------------------
# Device setup
# -----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _unwrap(m):
    """Unwrap torch.compile'd model if needed."""
    return getattr(m, "_orig_mod", m)


def _nparams(m, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in m.parameters()
               if (not trainable_only or p.requires_grad))


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.2f} M"
    if n >= 1_000:
        return f"{n/1e3:.2f} K"
    return str(n)


def _seed_everything(seed: int = 1337) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


def _safe_log10(x) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(x, float), 1e-12))


def _autocast_ctx(enabled: bool = True):
    return torch.amp.autocast(device_type="cuda", enabled=enabled and device.type == "cuda")


# -----------------------------------------------------------------------
# TrainSpec
# -----------------------------------------------------------------------
class TrainSpec:
    """
    Three-phase serialized spectral emulator trainer.

    All keyword arguments can be passed to __init__; the most important are
    listed in the Parameters section.  Run ``trainer.run()`` to execute all
    three phases in sequence.

    Parameters
    ----------
    modpath : str
        Path to directory (or single file) containing the spectral HDF5 grid.
    output : str
        Output HDF5 filename.  Each phase checkpoint is written here; the
        final file contains the phase-3 best weights.
    wave_range : (float, float), optional
        (lo_Å, hi_Å) wavelength window passed to ReadSpec.
    R : float, optional
        Resolving power for geometric resampling.
    dlambda : float, optional
        Constant Δλ resampling step (Å). Mutually exclusive with R.
    pixels_per_resel : float, default 3.0
    rebin_mode : str, default 'interp'
    H1, H2, H3 : int, default 512
        Hidden widths of the stellar head f0.
    W_k : int, default 128
        Hidden width of the extinction head khat.
    W_resid : int, default 256
        Hidden width of the residual head (two layers).
    n_epochs_p1/p2/p3 : int
        Per-phase epoch budgets.
    batchsize : int, default 128
    lr_p1/p2/p3 : float
        Peak learning rates per phase.  Defaults: 1e-3 / 3e-4 / 1e-4.
    eta_min : float, default 1e-7
        Cosine schedule floor (shared across phases).
    warmup_frac : float, default 0.05
        Fraction of each phase used for linear LR warm-up.
    warm_freeze_epochs : int, default 50
        Epochs at the start of phase 2 and 3 during which previously trained
        heads are frozen so newly activated heads can initialise.
    early_stopping : bool, default True
    early_stopping_patience : int, default 150
    early_stopping_min_delta : float, default 1e-5
    grad_accum_steps : int, default 1
    weight_by_logvar : bool, default True
        Weight loss by 1/(σ²_log + ε) per wavelength pixel.
        NOTE: This upweights pixels with low log-variance (continuum).
        Set False and use weight_by_logvar_inverse=False to upweight lines.
    weight_by_lines : bool, default False
        If True, weight by σ²_log (i.e. upweight high-variance line pixels
        instead of the continuum).  Overrides weight_by_logvar if both True.
    use_frac_loss : bool, default True
        Add a fractional-error term (MSE on f_pred/f_true - 1) alongside
        the log-space Huber loss.
    frac_alpha : float, default 0.5
        Mix weight: total_loss = (1-α)*log_huber + α*frac_mse.
    frac_clip_dex : float, default 4.0
        Clip |Δlog| before computing fractional error to prevent blow-up.
    coeff_loss_weight : float, default 0.05
        Weight of the PCA coefficient alignment auxiliary loss (only active
        when a PCA basis is used).
    split_seed : int, default 1337
    trainper : float, default 0.9
    num_workers : int, default 0
    plotevery : int, default 25
    plotdir : str, default './plots/'
    checkpointdir: str, default './checkpoints/'
    parrange : dict, optional
        Parameter range filters passed to ReadSpec.
    extinction_mode_p1 : str, default 'none'
    extinction_mode_p2/p3 : str, default 'sample'
    restartfile_p1/p2/p3 : str, optional
        Checkpoint files to restart each phase from (skips training for that
        phase if provided and resume_from_restart=True).
    resume_from_restart : bool, default False
    """

    # ----------------------------------------------------------------
    # __init__
    # ----------------------------------------------------------------
    def __init__(self, **cfg):
        print(f"[TrainSpec] Initialising at {datetime.now()}")

        # -- reproducibility --
        self.split_seed = cfg.get("split_seed", 1337)

        # -- data --
        self.modpath      = cfg.get("modpath", "./grid/h5/")
        self.wave_range   = cfg.get("wave_range", None)
        self.dlambda      = cfg.get("dlambda", None)
        self.R            = cfg.get("R", None)
        self.px_per_resel = cfg.get("pixels_per_resel", 3.0)
        self.rebin_mode   = cfg.get("rebin_mode", "interp")
        self.trainper     = cfg.get("trainper", 0.9)
        self.num_workers  = cfg.get("num_workers", 0)
        self.norm_outputs = False   # always train in raw flux / log space
        self.parrange     = cfg.get("parrange", {
            "logt": [3.0, 4.7],
            "logg": [-2.0, 6.0],
            "feh":  [-5.0, 1.0],
            "afe":  [-0.2, 0.6],
            "av":   [0.0, 20.0],
            "rv":   [2.0, 6.0],
        })

        # -- architecture --
        self.H1      = cfg.get("H1",      512)
        self.H2      = cfg.get("H2",      512)
        self.H3      = cfg.get("H3",      512)
        self.W_k     = cfg.get("W_k",     128)
        self.W_resid = cfg.get("W_resid", 256)

        # -- training --
        self.n_epochs_p1  = cfg.get("n_epochs_p1",  5000)
        self.n_epochs_p2  = cfg.get("n_epochs_p2",  3000)
        self.n_epochs_p3  = cfg.get("n_epochs_p3",  2000)
        self.batchsize    = cfg.get("batchsize", 128)
        self.lr_p1        = cfg.get("lr_p1", 1e-3)
        self.lr_p2        = cfg.get("lr_p2", 3e-4)
        self.lr_p3        = cfg.get("lr_p3", 1e-4)
        self.eta_min      = cfg.get("eta_min", 1e-7)
        self.warmup_frac  = cfg.get("warmup_frac", 0.05)
        self.warm_freeze_epochs = int(cfg.get("warm_freeze_epochs", 50))
        self.grad_accum_steps   = int(cfg.get("grad_accum_steps", 1))

        # -- loss --
        self.weight_by_logvar  = bool(cfg.get("weight_by_logvar", True))
        self.weight_by_lines   = bool(cfg.get("weight_by_lines", False))
        self.weight_eps        = float(cfg.get("weight_eps", 1e-7))
        self.use_frac_loss     = bool(cfg.get("use_frac_loss", True))
        self.frac_alpha        = float(cfg.get("frac_alpha", 0.5))
        self.frac_clip_dex     = float(cfg.get("frac_clip_dex", 4.0))
        self.coeff_loss_weight = float(cfg.get("coeff_loss_weight", 0.05))

        # -- early stopping --
        self.early_stopping          = bool(cfg.get("early_stopping", True))
        self.early_stopping_patience = int(cfg.get("early_stopping_patience", 150))
        self.early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 1e-5))

        # -- I/O --
        self.outfilename  = cfg.get("output", "TRAIN_SPEC_OUT.h5")
        self.plotevery    = cfg.get("plotevery", 25)
        self.plotdir      = cfg.get("plotdir", "./plots/")
        os.makedirs(self.plotdir, exist_ok=True)
        self.logplot      = cfg.get("logplot", True)
        self.checkpointdir = cfg.get("checkpointdir", "./checkpoints/")
        os.makedirs(self.checkpointdir, exist_ok=True)

        # per-phase extinction modes
        self.ext_mode_p1 = cfg.get("extinction_mode_p1", "none")
        self.ext_mode_p2 = cfg.get("extinction_mode_p2", "sample")
        self.ext_mode_p3 = cfg.get("extinction_mode_p3", "sample")
        self.fixed_av    = cfg.get("fixed_av", 0.0)
        self.fixed_rv    = cfg.get("fixed_rv", 3.1)

        # restart / resume
        self.restartfile_p1      = cfg.get("restartfile_p1", None)
        self.restartfile_p2      = cfg.get("restartfile_p2", None)
        self.restartfile_p3      = cfg.get("restartfile_p3", None)
        self.resume_from_restart = cfg.get("resume_from_restart", False)

        # populated after dataset construction
        self.label_i_p1: list = []
        self.label_i_p2: list = []
        self.label_i_p3: list = []
        self.label_o:    list = []
        self.vmic_is_fixed: bool = False

    # ----------------------------------------------------------------
    # Dataset construction
    # ----------------------------------------------------------------
    def _build_datasets(
        self,
        extinction_mode: str,
        label_i: list,
        inputs_have_avrv: bool,
    ):
        """Build a matched train/valid ReadSpec pair with a global seeded split."""

        common = dict(
            modpath        = self.modpath,
            wave_range     = self.wave_range,
            dlambda        = self.dlambda,
            R              = self.R,
            pixels_per_resel = self.px_per_resel,
            rebin_mode     = self.rebin_mode,
            norm           = self.norm_outputs,
            use_norm_from_h5 = True,
            returntorch    = True,
            trainpercentage = 1.0,
            parrange       = self.parrange,
            label_i        = label_i,
            split_seed     = self.split_seed,
        )

        # Anchor pass to get the full index universe
        anchor = readKorg.ReadSpec(
            **common,
            type            = "train",
            extinction_mode = extinction_mode,
            fixed_av        = self.fixed_av,
            fixed_rv        = self.fixed_rv,
            split           = None,
        )

        # Detect vmic on first construction
        if not hasattr(self, "_vmic_detected"):
            self.vmic_is_fixed    = anchor.vmic_is_fixed
            self._vmic_fixed_val  = anchor.vmic_fixed_value
            self._vmic_detected   = True
            if self.vmic_is_fixed:
                print(f"[TrainSpec] Fixed-vmic grid detected "
                      f"(vmic={self._vmic_fixed_val:.4f}). "
                      f"'vmic' will be excluded from all label_i.")

        # Global permutation split
        si      = anchor.split_indices
        all_idx = np.concatenate([si["train"], si["valid"], si["test"]]).astype(int)
        rng     = np.random.RandomState(self.split_seed)
        perm    = rng.permutation(all_idx)
        cut     = int(self.trainper * perm.size)
        split_dict = {
            "train": np.sort(perm[:cut]),
            "valid": np.sort(perm[cut:]),
            "test":  np.array([], dtype=int),
        }

        ds_train = readKorg.ReadSpec(
            **common,
            type            = "train",
            extinction_mode = extinction_mode,
            fixed_av        = self.fixed_av,
            fixed_rv        = self.fixed_rv,
            split           = split_dict,
        )
        ds_valid = readKorg.ReadSpec(
            **common,
            type            = "valid",
            # Always use fixed Av=0 for validation to get a clean intrinsic
            # metric regardless of which phase we're in.
            extinction_mode = ("fixed" if extinction_mode != "none" else "none"),
            fixed_av        = 0.0,
            fixed_rv        = self.fixed_rv,
            split           = split_dict,
        )

        return ds_train, ds_valid, split_dict, anchor

    # ----------------------------------------------------------------
    # PCA basis
    # ----------------------------------------------------------------
    def _build_basis(self, ds_flat, K: Optional[int]) -> tuple:
        """Compute PCA basis in log10 flux space on the training split.

        Returns (B, mu_log, sd_log) where:
          B      — (K, L) basis tensor or None
          mu_log — (L,) mean log10 spectrum
          sd_log — (L,) std  log10 spectrum
        """
        L   = len(ds_flat.label_o)
        cap = min(20_000, len(ds_flat))

        # Use a DataLoader for faster collection instead of item-by-item loop
        loader = DataLoader(
            XYFromFlat(ds_flat),
            batch_size  = 512,
            sampler     = SequentialSampler(XYFromFlat(ds_flat)),
            num_workers = 0,
            drop_last   = False,
        )
        chunks = []
        n_collected = 0
        with torch.no_grad():
            for _, yb in loader:
                ylog = torch.log10(torch.clamp(yb.float(), min=1e-12))
                chunks.append(ylog)
                n_collected += yb.size(0)
                if n_collected >= cap:
                    break
        X = torch.cat(chunks, dim=0)[:cap]   # (cap, L)

        mu_log = X.mean(dim=0)               # (L,)
        sd_log = X.std(dim=0, unbiased=False)
        Xm     = X - mu_log.unsqueeze(0)

        if K is not None and K > 0:
            K_eff = min(int(K), L)
            if K_eff != K:
                print(f"[basis] Reducing basis_K {K} → {K_eff} (L={L}).")
            # Randomised SVD is much cheaper than full SVD for K << L
            try:
                from sklearn.utils.extmath import randomized_svd
                _, _, Vt = randomized_svd(
                    Xm.numpy(), n_components=K_eff, random_state=self.split_seed
                )
                B = torch.tensor(Vt, dtype=torch.float32)      # (K_eff, L)
            except ImportError:
                # Fall back to torch SVD if sklearn unavailable
                print("[basis] sklearn not found; falling back to torch.linalg.svd "
                      "(may be slow for large L).")
                _, _, Vt = torch.linalg.svd(Xm, full_matrices=False)
                B = Vt[:K_eff, :]
            return B.cpu(), mu_log.cpu(), sd_log.cpu()
        else:
            return None, mu_log.cpu(), sd_log.cpu()

    # ----------------------------------------------------------------
    # Optimiser + scheduler factory
    # ----------------------------------------------------------------
    @staticmethod
    def _make_opt_sched(model, lr: float, n_epochs: int,
                        warmup_frac: float, eta_min: float):
        """AdamW with weight decay + linear warm-up + cosine decay."""
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("bias") or "ln" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        opt = torch.optim.AdamW(
            [{"params": decay,    "weight_decay": 1e-4},
             {"params": no_decay, "weight_decay": 0.0}],
            lr    = lr,
            betas = (0.9, 0.999),
            fused = (device.type == "cuda"),
        )

        warmup = max(1, int(warmup_frac * n_epochs))
        warm   = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0 / warmup, end_factor=1.0, total_iters=warmup
        )
        cos    = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, n_epochs - warmup), eta_min=eta_min
        )
        sched  = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warm, cos], milestones=[warmup]
        )
        return opt, sched

    # ----------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------
    def _compute_loss(
        self,
        yhat_log: torch.Tensor,
        ylog:     torch.Tensor,
        w_pix:    Optional[torch.Tensor],
        model,
        is_train: bool = True,
    ) -> tuple:
        """Compute combined loss and coefficient alignment term.

        Returns (total_loss, data_loss, coeff_loss) as scalars.
        """
        huber = nn.SmoothL1Loss(reduction="mean")

        # 1) log-space Huber
        if w_pix is None:
            log_huber = huber(yhat_log, ylog)
        else:
            per_el    = F.smooth_l1_loss(yhat_log, ylog, reduction="none")
            log_huber = (per_el * w_pix).mean()

        # 2) fractional error in log space
        if self.use_frac_loss:
            dlog   = torch.clamp(yhat_log - ylog,
                                 min=-self.frac_clip_dex, max=self.frac_clip_dex)
            ratio  = torch.pow(10.0, dlog)
            frac   = ratio - 1.0
            if w_pix is None:
                frac_mse = (frac * frac).mean()
            else:
                frac_mse = ((frac * frac) * w_pix).mean()
            data_loss = ((1.0 - self.frac_alpha) * log_huber
                         + self.frac_alpha * frac_mse)
        else:
            data_loss = log_huber

        # 3) coefficient alignment (only when basis head present)
        coeff_loss = torch.tensor(0.0, device=yhat_log.device)
        m = _unwrap(model)
        if m.has_basis:
            mu  = (m.mu_log if (m.mu_log is not None
                                and isinstance(m.mu_log, torch.Tensor))
                   else 0.0)
            Bt  = m.f0.B.B.t()         # (L, K)
            z_t = (ylog     - mu) @ Bt
            z_h = (yhat_log - mu) @ Bt
            coeff_loss = F.mse_loss(z_h, z_t)

        total = data_loss + self.coeff_loss_weight * coeff_loss
        return total, data_loss, coeff_loss

    # ----------------------------------------------------------------
    # Single-phase training loop
    # ----------------------------------------------------------------
    def _run_phase(
        self,
        phase:            int,
        model:            nn.Module,
        train_loader:     DataLoader,
        valid_loader:     DataLoader,
        n_epochs:         int,
        lr:               float,
        w_pix:            Optional[torch.Tensor],
        phase_label:      str,
        restartfile:      Optional[str] = None,
        warm_freeze:      bool = False,
    ) -> nn.Module:
        """Run one training phase; returns the model with best weights loaded."""

        print(f"\n{'='*60}")
        print(f"  Phase {phase}: {phase_label}")
        print(f"  epochs={n_epochs}, lr={lr:.1e}, "
              f"warm_freeze={self.warm_freeze_epochs if warm_freeze else 0}")
        print(f"  Trainable params: {_fmt(_nparams(model))}")
        print(f"{'='*60}")

        # Optional restart: load weights and (if resuming) skip this phase
        start_epoch = 0
        best_val    = float("inf")
        if restartfile and os.path.isfile(restartfile):
            print(f"  Loading checkpoint: {restartfile}")
            load_state_dict_from_h5(_unwrap(model), restartfile,
                                    group="model", strict=True,
                                    dtype=torch.float32)
            if self.resume_from_restart:
                try:
                    with h5py.File(restartfile, "r") as h5:
                        g           = h5.get("meta", {})
                        start_epoch = int(g.attrs.get("epochs_trained", 0))
                        best_val    = float(g.attrs.get("best_valid", np.inf))
                    print(f"  Resuming from epoch {start_epoch}, "
                          f"best_valid={best_val:.4e}")
                    if start_epoch >= n_epochs:
                        print(f"  Phase already complete ({start_epoch} ≥ {n_epochs}). "
                              f"Skipping.")
                        return model
                except Exception as e:
                    print(f"  Warning: could not read restart meta: {e}")
                    start_epoch = 0
                    best_val    = float("inf")

        # Warm-freeze: freeze previously trained heads at phase start
        # We do this once here rather than per-epoch to keep optimizer
        # moment buffers consistent.
        if warm_freeze and phase == 2:
            _unwrap(model).freeze_f0(True)
            print(f"  f0 frozen for first {self.warm_freeze_epochs} epochs.")
        if warm_freeze and phase == 3:
            _unwrap(model).freeze_f0(True)
            _unwrap(model).freeze_extinction(True)
            print(f"  f0 + khat frozen for first {self.warm_freeze_epochs} epochs.")

        opt, sched = self._make_opt_sched(model, lr, n_epochs,
                                          self.warmup_frac, self.eta_min)
        scaler     = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        # -- loss tracking --
        train_means, train_stds, train_meds = [], [], []
        valid_means, valid_stds, valid_meds = [], [], []

        fig, axes = plt.subplots(3, 1, figsize=(7, 10), layout="constrained")
        for ax in axes:
            ax.set_xlim(0, n_epochs)
        axes[0].set_ylabel("log₁₀(loss mean)")
        axes[1].set_ylabel("log₁₀(loss std)")
        axes[2].set_ylabel("log₁₀(loss median)")
        axes[2].set_xlabel("Epoch")

        es_counter = 0

        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()

            # Unfreeze heads after warm-in period
            if warm_freeze:
                if epoch == self.warm_freeze_epochs:
                    _unwrap(model).freeze_f0(False)
                    if phase == 3:
                        _unwrap(model).freeze_extinction(False)
                    # Rebuild optimiser so newly unfrozen params get fresh moments
                    opt, sched = self._make_opt_sched(
                        model, lr, n_epochs - epoch,
                        self.warmup_frac, self.eta_min
                    )
                    print(f"  Epoch {epoch+1}: heads unfrozen; "
                          f"trainable params now {_fmt(_nparams(model))}.")

            # -- train --
            model.train()
            losses = []
            opt.zero_grad(set_to_none=True)

            for step, (xb, yb) in enumerate(train_loader, start=1):
                xb   = xb.to(device, non_blocking=True)
                yb   = yb.to(device, non_blocking=True)
                ylog = torch.log10(torch.clamp(yb, min=1e-12))

                with _autocast_ctx(device.type == "cuda"):
                    yhat_log = model(xb)
                    loss, _, _ = self._compute_loss(yhat_log, ylog, w_pix, model)
                    loss = loss / max(1, self.grad_accum_steps)

                scaler.scale(loss).backward()

                if step % self.grad_accum_steps == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                losses.append(loss.detach().item() * self.grad_accum_steps)

            # Final partial accumulation step
            if (step % self.grad_accum_steps) != 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            sched.step()

            train_means.append(float(np.mean(losses)))
            train_stds.append(float(np.std(losses)))
            train_meds.append(float(np.median(losses)))

            # -- validate --
            model.eval()
            v_losses = []
            with torch.inference_mode(), _autocast_ctx(device.type == "cuda"):
                for xb, yb in valid_loader:
                    xb   = xb.to(device, non_blocking=True)
                    yb   = yb.to(device, non_blocking=True)
                    ylog = torch.log10(torch.clamp(yb, min=1e-12))
                    yhat_log = model(xb)
                    v, _, _ = self._compute_loss(yhat_log, ylog, w_pix, model,
                                                 is_train=False)
                    v_losses.append(v.item())

            val_mean = float(np.mean(v_losses)) if v_losses else float("inf")
            val_std  = float(np.std(v_losses))  if v_losses else 0.0
            val_med  = float(np.median(v_losses)) if v_losses else val_mean

            valid_means.append(val_mean)
            valid_stds.append(val_std)
            valid_meds.append(val_med)

            # -- checkpoint best --
            if val_mean < best_val - self.early_stopping_min_delta:
                best_val   = val_mean
                es_counter = 0
                save_state_dict_to_h5(
                    _unwrap(model).state_dict(),
                    self.outfilename, group="model", compression="gzip"
                )
                save_meta_to_h5(
                    self.outfilename,
                    n_inputs      = _unwrap(model).d_full,
                    n_outputs     = _unwrap(model).L,
                    nn_type       = "SpectralMLP_v3",
                    best_valid    = float(best_val),
                    epochs_trained = int(epoch + 1),
                    phase         = int(phase),
                    date          = str(datetime.now()),
                    R             = float(self.R or 0.0),
                    pixels_per_resel = float(self.px_per_resel),
                )
                # Store mu_log for non-PyTorch consumers
                try:
                    with h5py.File(self.outfilename, "a") as h5:
                        if "mu_log" in h5:
                            del h5["mu_log"]
                        m_log = _unwrap(model).mu_log
                        if m_log is not None:
                            h5.create_dataset(
                                "mu_log",
                                data=m_log.detach().cpu().numpy(),
                                compression="gzip",
                            )
                except Exception:
                    pass
            else:
                es_counter += 1

            # -- early stopping --
            if self.early_stopping and es_counter >= self.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {es_counter} epochs).")
                break

            # -- logging --
            if epoch % 25 == 0 or epoch == n_epochs - 1:
                print(
                    f"  Ep {epoch+1:>5}/{n_epochs}  "
                    f"train={math.log10(max(train_means[-1], 1e-12)):+.4f}  "
                    f"valid={math.log10(max(val_mean,         1e-12)):+.4f}  "
                    f"best={math.log10(max(best_val,          1e-12)):+.4f}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  "
                    f"t={time.time()-t0:.1f}s"
                )

            # -- loss plot --
            if epoch % self.plotevery == 0 or epoch == n_epochs - 1:
                bx = np.arange(len(train_means))
                for ii, (tr, va) in enumerate(
                    zip([train_means, train_stds, train_meds],
                        [valid_means, valid_stds, valid_meds])
                ):
                    ax = axes[ii]
                    ax.plot(bx, _safe_log10(tr), c="C0", lw=0.8,
                            label="train" if (epoch == 0 and ii == 0) else None)
                    ax.plot(bx, _safe_log10(va), c="C3", lw=0.8,
                            label="valid" if (epoch == 0 and ii == 0) else None)
                    ax.set_xlim(0, epoch + 1)
                    all_v = np.concatenate([_safe_log10(tr), _safe_log10(va)])
                    finite = all_v[np.isfinite(all_v)]
                    if len(finite):
                        lo5, hi95 = np.percentile(finite, [5, 95])
                        margin    = 0.1 * max(hi95 - lo5, 1e-3)
                        ax.set_ylim(lo5 - margin, hi95 + margin)
                if epoch == 0:
                    axes[0].legend(loc="best", fontsize=9)
                stem = os.path.splitext(
                    os.path.basename(self.outfilename))[0]
                fig.savefig(
                    f"{self.plotdir}/{stem}_phase{phase}_loss.png", dpi=150
                )

        plt.close(fig)

        # Reload best weights so the returned model is at its best val point
        if os.path.isfile(self.outfilename):
            load_state_dict_from_h5(_unwrap(model), self.outfilename,
                                    group="model", strict=True,
                                    dtype=torch.float32)
            print(f"  Best weights reloaded (best_val={best_val:.4e}).")

        del train_loader, valid_loader, opt, sched, scaler
        gc.collect()

        return model

    # ----------------------------------------------------------------
    # run()
    # ----------------------------------------------------------------
    def run(self, basis_K: Optional[int] = None, dryrun: bool = False):
        """Execute all three training phases in sequence.

        Parameters
        ----------
        basis_K : int, optional
            Number of PCA components for the low-rank basis head.
            None → direct pixel-by-pixel prediction (no basis).
            Recommended: 50–200 for L > 2000.
        dryrun : bool
            If True, build datasets and model but skip training.
        """
        _seed_everything(self.split_seed)

        # ============================================================
        # Phase 1 setup: intrinsic-only, no av/rv in inputs
        # ============================================================
        print("\n[Phase 1] Building intrinsic datasets ...")

        # Determine label_i for phase 1 (no av/rv)
        # vmic inclusion is decided after the first ReadSpec construction
        # (which auto-detects fixed vmic); we pass a provisional list and
        # let ReadSpec silently drop vmic if fixed.
        provisional_p1 = ["logt", "logg", "feh", "afe", "vmic"]
        ds_p1_flat, ds_p1_valid_flat, split, anchor = self._build_datasets(
            extinction_mode  = self.ext_mode_p1,
            label_i          = provisional_p1,
            inputs_have_avrv = False,
        )
        # After construction, label_i reflects what ReadSpec actually used
        self.label_i_p1 = list(ds_p1_flat.label_i)
        self.label_o    = list(ds_p1_flat.label_o)

        L      = len(self.label_o)
        d_phys = len(self.label_i_p1)   # no av/rv in phase 1
        d_full_p1 = d_phys              # same for phase 1

        print(f"  L={L}, d_phys={d_phys}, vmic_fixed={self.vmic_is_fixed}")

        # Label_i for phases 2/3 adds av, rv
        self.label_i_p2 = self.label_i_p1 + ["av", "rv"]
        self.label_i_p3 = self.label_i_p2
        d_full = d_phys + 2

        # ---- PCA basis ----
        print("  Computing PCA basis on training split ...")
        basis_B, mu_log, sd_log = self._build_basis(ds_p1_flat, K=basis_K)
        if basis_B is not None:
            # Save basis_B into the checkpoint for later reconstruction
            with h5py.File(self.outfilename, "a") as h5:
                if "basis_B" in h5:
                    del h5["basis_B"]
                h5.create_dataset("basis_B",
                                data=basis_B.numpy(),
                                compression="gzip")
            print(f"  Basis: K={basis_B.shape[0]}, L={basis_B.shape[1]}")

        # ---- pixel weights from log-variance ----
        if (self.weight_by_logvar or self.weight_by_lines) and sd_log is not None:
            sd = torch.as_tensor(sd_log, dtype=torch.float32, device=device).view(-1)
            if self.weight_by_lines:
                # Upweight high-variance (line) pixels
                w_pix = sd * sd
            else:
                # Upweight low-variance (continuum) pixels
                w_pix = 1.0 / (sd * sd + self.weight_eps)
            w_pix = w_pix / w_pix.mean()
        else:
            w_pix = None

        # ---- save label / wavelength metadata ----
        with h5py.File(self.outfilename, "a") as h5:
            for key, val in [("label_i_p1", self.label_i_p1),
                              ("label_i_p2", self.label_i_p2),
                              ("label_o",    self.label_o)]:
                if key in h5:
                    del h5[key]
                h5.create_dataset(
                    key, data=np.array([s.encode("ascii") for s in val])
                )
            if "wavelengths_A" in h5:
                del h5["wavelengths_A"]
            h5.create_dataset(
                "wavelengths_A",
                data=np.asarray(ds_p1_flat.wavelengths_A, dtype=np.float64),
            )
            # Save basis_B unconditionally so checkpoints are self-contained
            if basis_B is not None:
                if "basis_B" in h5:
                    del h5["basis_B"]
                h5.create_dataset("basis_B",
                                data=basis_B.numpy(),
                                compression="gzip")
            
            g = h5.require_group("meta")
            g.attrs.update({
                "created":    str(datetime.now()),
                "trainper":   self.trainper,
                "batchsize":  self.batchsize,
                "modpath":    str(self.modpath),
                "vmic_fixed": int(self.vmic_is_fixed),
                "d_phys":     d_phys,
                "L":          L,
            })

        # ---- build model ----
        model = SpectralMLP_v3(
            d_phys             = d_phys,
            d_full             = d_full,
            L                  = L,
            H1                 = self.H1,
            H2                 = self.H2,
            H3                 = self.H3,
            W_k                = self.W_k,
            W_resid            = self.W_resid,
            basis_B            = basis_B,
            mu_log             = mu_log,
            include_extinction = False,   # phase 1: off
            include_resid      = False,   # phase 1: off
            inputs_have_avrv   = False,   # phase 1: no av/rv
            resid_gate_init    = -2.0,    # start residual lane silent
        ).to(device)

        print(f"  Model: {_fmt(model.n_params())} trainable params")

        if dryrun:
            print("[dryrun] Skipping training.")
            return model

        # ============================================================
        # Phase 1: train f0 (intrinsic, no extinction)
        # ============================================================
        def _make_loaders(ds_flat, ds_valid_flat, label_i):
            train_ds = XYFromFlat(ds_flat)
            valid_ds = XYFromFlat(ds_valid_flat)
            n_tr     = len(train_ds)
            n_va     = len(valid_ds)
            bs_tr    = min(self.batchsize, max(1, n_tr))
            bs_va    = min(self.batchsize, max(1, n_va))
            wif      = partial(_worker_init_fn, base_seed=self.split_seed)
            tl = DataLoader(
                train_ds,
                sampler      = RandomSampler(train_ds),
                batch_size   = bs_tr,
                num_workers  = self.num_workers,
                worker_init_fn = wif if self.num_workers > 0 else None,
                pin_memory   = (device.type == "cuda"),
                drop_last    = False,
                persistent_workers = False,
                prefetch_factor = None if self.num_workers == 0 else 2,
            )
            vl = DataLoader(
                valid_ds,
                sampler      = SequentialSampler(valid_ds),
                batch_size   = bs_va,
                num_workers  = self.num_workers,
                worker_init_fn = wif if self.num_workers > 0 else None,
                pin_memory   = (device.type == "cuda"),
                drop_last    = False,
                persistent_workers = False,
                prefetch_factor = None if self.num_workers == 0 else 2,
            )
            return tl, vl

        tl_p1, vl_p1 = _make_loaders(ds_p1_flat, ds_p1_valid_flat,
                                      self.label_i_p1)
        model = self._run_phase(
            phase        = 1,
            model        = model,
            train_loader = tl_p1,
            valid_loader = vl_p1,
            n_epochs     = self.n_epochs_p1,
            lr           = self.lr_p1,
            w_pix        = w_pix,
            phase_label  = "Intrinsic only (f0)",
            restartfile  = self.restartfile_p1,
            warm_freeze  = False,
        )
        # Save phase-1 checkpoint with clear name
        p1_ckpt = os.path.join(self.checkpointdir, self.outfilename.replace(".h5", "_phase1.h5"))
        import shutil
        shutil.copy2(self.outfilename, p1_ckpt)
        print(f"  Phase-1 checkpoint saved: {p1_ckpt}")

        # ============================================================
        # Phase 2: add extinction (khat)
        # ============================================================
        print("\n[Phase 2] Building extinction datasets ...")
        ds_p2_flat, ds_p2_valid_flat, _, _ = self._build_datasets(
            extinction_mode  = self.ext_mode_p2,
            label_i          = self.label_i_p2,
            inputs_have_avrv = True,
        )

        # Reconfigure model for phase 2: enable extinction, add av/rv to inputs
        model_p2 = SpectralMLP_v3(
            d_phys             = d_phys,
            d_full             = d_full,
            L                  = L,
            H1                 = self.H1,
            H2                 = self.H2,
            H3                 = self.H3,
            W_k                = self.W_k,
            W_resid            = self.W_resid,
            basis_B            = basis_B,
            mu_log             = mu_log,
            include_extinction = True,
            include_resid      = False,
            inputs_have_avrv   = True,
            resid_gate_init    = -2.0,
        ).to(device)

        # Transfer f0 weights from phase 1
        src = _unwrap(model).f0.state_dict()
        _unwrap(model_p2).f0.load_state_dict(src, strict=True)
        # Also transfer gates
        with torch.no_grad():
            _unwrap(model_p2).ext_gate.copy_(_unwrap(model).ext_gate)
            _unwrap(model_p2).res_gate.copy_(_unwrap(model).res_gate)
        del model

        tl_p2, vl_p2 = _make_loaders(ds_p2_flat, ds_p2_valid_flat,
                                      self.label_i_p2)
        model_p2 = self._run_phase(
            phase        = 2,
            model        = model_p2,
            train_loader = tl_p2,
            valid_loader = vl_p2,
            n_epochs     = self.n_epochs_p2,
            lr           = self.lr_p2,
            w_pix        = w_pix,
            phase_label  = "f0 + extinction (khat)",
            restartfile  = self.restartfile_p2,
            warm_freeze  = True,
        )
        p2_ckpt = os.path.join(self.checkpointdir, self.outfilename.replace(".h5", "_phase2.h5"))
        shutil.copy2(self.outfilename, p2_ckpt)
        print(f"  Phase-2 checkpoint saved: {p2_ckpt}")

        # ============================================================
        # Phase 3: add residual correction
        # ============================================================
        print("\n[Phase 3] Building residual-phase datasets ...")
        ds_p3_flat, ds_p3_valid_flat, _, _ = self._build_datasets(
            extinction_mode  = self.ext_mode_p3,
            label_i          = self.label_i_p3,
            inputs_have_avrv = True,
        )

        model_p3 = SpectralMLP_v3(
            d_phys             = d_phys,
            d_full             = d_full,
            L                  = L,
            H1                 = self.H1,
            H2                 = self.H2,
            H3                 = self.H3,
            W_k                = self.W_k,
            W_resid            = self.W_resid,
            basis_B            = basis_B,
            mu_log             = mu_log,
            include_extinction = True,
            include_resid      = True,
            inputs_have_avrv   = True,
            resid_gate_init    = -2.0,   # residual starts silent
        ).to(device)

        # Transfer f0 + khat + gates from phase 2
        src_sd = _unwrap(model_p2).state_dict()
        # Load everything that matches (all of f0 and khat; resid reinits at zero)
        _unwrap(model_p3).load_state_dict(src_sd, strict=False)
        del model_p2

        tl_p3, vl_p3 = _make_loaders(ds_p3_flat, ds_p3_valid_flat,
                                      self.label_i_p3)
        model_p3 = self._run_phase(
            phase        = 3,
            model        = model_p3,
            train_loader = tl_p3,
            valid_loader = vl_p3,
            n_epochs     = self.n_epochs_p3,
            lr           = self.lr_p3,
            w_pix        = w_pix,
            phase_label  = "f0 + khat + residual (all heads)",
            restartfile  = self.restartfile_p3,
            warm_freeze  = True,
        )
        p3_ckpt = os.path.join(self.checkpointdir, self.outfilename.replace(".h5", "_phase3.h5"))
        shutil.copy2(self.outfilename, p3_ckpt)
        print(f"  Phase-3 checkpoint saved: {p3_ckpt}")

        torch.cuda.empty_cache()
        print("\n[TrainSpec] All three phases complete.")
        gc.collect()
        return model_p3