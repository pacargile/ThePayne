"""testspec_v3.py

Diagnostic plots for a trained SpectralMLP_v3 emulator loaded from a
phase-3 HDF5 checkpoint.

Usage
-----
    from diagnose_emulator import TestSpec

    diag = TestSpec(
        checkpoint  = "./checkpoints/naD_R65k_run1_phase3.h5",
        modpath     = "./specgrid/",
        plotdir     = "./plots/",
        split_seed  = 1337,
        parrange    = {...},   # same ranges used during training
        n_test      = 2000,    # how many test spectra to evaluate
    )
    diag.run_all()

Individual plots can also be called separately, e.g.::

    diag.plot_spectral_residuals()
    diag.plot_residuals_vs_params()

Plots produced
--------------
1.  spectral_residuals.png       — true vs predicted spectra + residual envelope
2.  residuals_vs_params.png      — median |Δ| in dex vs each stellar parameter
3.  pca_coefficients.png         — z_true vs z_hat scatter for each PCA mode
4.  line_profile_accuracy.png    — Na D line core zoom for representative stars
5.  khat_extinction_curve.png    — learned k̂(λ) vs G23 reference for Rv grid
6.  hr_diagram_residuals.png     — (logt, logg) HR diagram coloured by RMS error
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Tuple

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from dust_extinction.parameter_averages import G23
from astropy import units as u

# local imports — adjust to your package layout
from Payne.utils import readKorg
from Payne.utils.readKorg import XYFromFlat
from Payne.utils.io_h5 import load_state_dict_from_h5
from Payne.train.NNmodels_new_v3 import SpectralMLP_v3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------
# Plotting style defaults
# -----------------------------------------------------------------------
STYLE = {
    "figure.dpi":        150,
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

PARAM_LABELS = {
    "logt": r"$\log\,T_{\rm eff}$",
    "logg": r"$\log\,g$",
    "feh":  r"$[\mathrm{Fe/H}]$",
    "afe":  r"$[\alpha/\mathrm{Fe}]$",
    "vmic": r"$\xi_t$ (km/s)",
    "av":   r"$A_V$",
    "rv":   r"$R_V$",
}


# -----------------------------------------------------------------------
# Checkpoint loader
# -----------------------------------------------------------------------
def load_checkpoint(checkpoint: str) -> dict:
    """Read all datasets and meta attributes from the phase-3 HDF5 checkpoint.

    Returns a dict with keys:
        wavelengths_A, label_i_p1, label_i_p2, label_o,
        mu_log (or None), meta (dict of attrs),
        model_weights (OrderedDict), basis_B (or None)
    """
    out = {}
    with h5py.File(checkpoint, "r") as h5:
        out["wavelengths_A"] = np.array(h5["wavelengths_A"][()], dtype=np.float64)
        out["label_i_p1"]    = [s.decode() for s in h5["label_i_p1"][()]]
        out["label_i_p2"]    = [s.decode() for s in h5["label_i_p2"][()]]
        out["label_o"]       = [s.decode() for s in h5["label_o"][()]]
        out["mu_log"]        = (np.array(h5["mu_log"][()], dtype=np.float32)
                                if "mu_log" in h5 else None)
        # meta attrs
        out["meta"] = dict(h5["meta"].attrs) if "meta" in h5 else {}
        # model weights stored under group "model"
        # load_state_dict_from_h5 handles this; just note the path
        out["checkpoint_path"] = checkpoint
        out["basis_B"] = (np.array(h5["basis_B"][()], dtype=np.float32)
                  if "basis_B" in h5 else None)
    return out


def build_model_from_checkpoint(
    checkpoint:  str,
    basis_B:     Optional[torch.Tensor] = None,
    H1: int = 512, H2: int = 512, H3: int = 512,
    W_k: int = 128, W_resid: int = 256,
) -> Tuple[SpectralMLP_v3, dict]:
    """Reconstruct and load a SpectralMLP_v3 from a phase-3 checkpoint.

    The basis matrix *basis_B* must be the same (K, L) tensor used during
    training. If you stored it separately (e.g. as a numpy .npy file) pass
    it here. If None the model is built without a basis (direct pixel head).

    Returns (model, ckpt_dict) where ckpt_dict contains wavelengths, labels, etc.
    """
    ckpt = load_checkpoint(checkpoint)

    # Use basis from checkpoint if not explicitly provided
    if basis_B is None and ckpt.get("basis_B") is not None:
        basis_B = torch.tensor(ckpt["basis_B"], dtype=torch.float32)

    meta = ckpt["meta"]

    d_phys = int(meta.get("d_phys", len(ckpt["label_i_p1"])))
    L      = int(meta.get("L",      len(ckpt["label_o"])))
    d_full = d_phys + 2   # +av, +rv (phase-3 model always has avrv)

    mu_log = (torch.tensor(ckpt["mu_log"], dtype=torch.float32)
              if ckpt["mu_log"] is not None else None)

    model = SpectralMLP_v3(
        d_phys             = d_phys,
        d_full             = d_full,
        L                  = L,
        H1                 = H1, H2=H2, H3=H3,
        W_k                = W_k,
        W_resid            = W_resid,
        basis_B            = basis_B,
        mu_log             = mu_log,
        include_extinction = True,
        include_resid      = True,
        inputs_have_avrv   = True,
    ).to(device)

    load_state_dict_from_h5(model, checkpoint, group="model",
                            strict=True, dtype=torch.float32)
    model.eval()
    return model, ckpt


# -----------------------------------------------------------------------
# Main diagnostics class
# -----------------------------------------------------------------------
class TestSpec:
    """
    Load a trained SpectralMLP_v3 from an HDF5 checkpoint and generate a
    suite of diagnostic plots on a held-out test set drawn from ReadSpec.

    Parameters
    ----------
    checkpoint : str
        Path to the phase-3 HDF5 checkpoint file.
    modpath : str
        Path to the spectral grid directory (same as used for training).
    basis_B : np.ndarray of shape (K, L), optional
        PCA basis matrix used during training. If you saved it separately
        (e.g. via np.save) load it and pass it here.
        Pass None to use the direct pixel head (no basis).
    plotdir : str, default './plots/'
        Directory where PNG diagnostics are saved.
    outputtag: str, default ''
        Optional tag to append to plot filenames for disambiguation.
    split_seed : int, default 1337
        Must match the seed used during training for reproducible splits.
    trainper : float, default 0.9
        Must match the value used during training.
    parrange : dict, optional
        Parameter range filters — should match those used during training.
    n_test : int, default 2000
        Number of test spectra to evaluate. Capped at the test-set size.
    wave_range : (float, float), optional
        Wavelength window to pass to ReadSpec. Read from checkpoint if not given.
    R : float, optional
        Resolving power. Read from checkpoint meta if not given.
    pixels_per_resel : float, default 3.0
    H1, H2, H3 : int, default 512
    W_k : int, default 128
    W_resid : int, default 256
    batch_size : int, default 512
        Batch size for inference. Larger = faster on GPU.
    seed : int, default 42
        RNG seed for reproducible sub-sampling of the test set.
    verbose : bool, default True
    """

    def __init__(
        self,
        checkpoint:      str,
        modpath:         str,
        basis_B:          Optional[np.ndarray] = None,
        plotdir:         str = "./plots/",
        outputtag:       str = "",
        split_seed:      int = 1337,
        trainper:        float = 0.9,
        parrange:        Optional[dict] = None,
        n_test:          int = 2000,
        wave_range:      Optional[Tuple[float, float]] = None,
        R:               Optional[float] = None,
        pixels_per_resel: float = 3.0,
        H1: int = 512, H2: int = 512, H3: int = 512,
        W_k: int = 128, W_resid: int = 256,
        batch_size:      int = 512,
        seed:            int = 42,
        verbose:         bool = True,
    ):
        self.checkpoint  = checkpoint
        self.modpath     = modpath
        self.plotdir     = plotdir
        self.outputtag   = outputtag
        self.split_seed  = split_seed
        self.trainper    = trainper
        self.parrange    = parrange
        self.n_test      = n_test
        self.batch_size  = batch_size
        self.seed        = seed
        self.verbose     = verbose
        os.makedirs(plotdir, exist_ok=True)

        # ---- load checkpoint metadata ----
        self._ckpt = load_checkpoint(checkpoint)
        meta       = self._ckpt["meta"]

        self.wavelengths_A = self._ckpt["wavelengths_A"]
        self.label_i_p1    = self._ckpt["label_i_p1"]
        self.label_i_p2    = self._ckpt["label_i_p2"]
        self.label_o       = self._ckpt["label_o"]
        self.L             = len(self.label_o)

        # Fall back to checkpoint meta for R / wave_range if not supplied
        self.R               = R or float(meta.get("R", 0.0)) or None
        self.pixels_per_resel = pixels_per_resel
        # wave_range: infer from stored wavelengths if not given
        if wave_range is not None:
            self.wave_range = wave_range
        else:
            self.wave_range = (float(self.wavelengths_A[0]),
                               float(self.wavelengths_A[-1]))

        # ---- load model ----
        if self.verbose:
            print(f"[Diagnostics] Loading model from {checkpoint}")
        basis_tensor = (torch.tensor(np.asarray(basis_B, dtype=np.float32))
                        if basis_B is not None else None)

        self.model, _ = build_model_from_checkpoint(
            checkpoint, basis_tensor,
            H1=H1, H2=H2, H3=H3, W_k=W_k, W_resid=W_resid,
            )

        if self.verbose:
            n = sum(p.numel() for p in self.model.parameters())
            print(f"[Diagnostics] Model loaded: {n:,} parameters")

        # ---- build test dataset ----
        if self.verbose:
            print(f"[Diagnostics] Building test dataset from {modpath}")
        self._build_test_data()

    # ----------------------------------------------------------------
    # Dataset construction
    # ----------------------------------------------------------------
    def _build_test_data(self):
        """Build the held-out test split using the same seed as training."""

        # Anchor pass to recover the global split indices
        anchor = readKorg.ReadSpec(
            modpath          = self.modpath,
            wave_range       = self.wave_range,
            R                = self.R,
            pixels_per_resel = self.pixels_per_resel,
            rebin_mode       = "interp",
            norm             = False,
            use_norm_from_h5 = True,
            returntorch      = True,
            type             = "train",
            trainpercentage  = 1.0,
            parrange         = self.parrange,
            label_i          = self.label_i_p2,   # full label set (with av/rv)
            extinction_mode  = "none",
            split_seed       = self.split_seed,
            split            = None,
        )

        # Reconstruct the same global split used during training
        si      = anchor.split_indices
        all_idx = np.concatenate([si["train"], si["valid"], si["test"]]).astype(int)
        rng     = np.random.RandomState(self.split_seed)
        perm    = rng.permutation(all_idx)
        cut     = int(self.trainper * perm.size)
        split_dict = {
            "train": np.sort(perm[:cut]),
            "valid": np.sort(perm[cut:]),
            # test = valid for diagnostic purposes; there is no separate
            # held-out test set by default — use valid as a clean proxy.
            "test":  np.sort(perm[cut:]),
        }

        # Build intrinsic test dataset (no extinction, av=0)
        # so residuals reflect only the stellar head accuracy
        ds_test = readKorg.ReadSpec(
            modpath          = self.modpath,
            wave_range       = self.wave_range,
            R                = self.R,
            pixels_per_resel = self.pixels_per_resel,
            rebin_mode       = "interp",
            norm             = False,
            use_norm_from_h5 = True,
            returntorch      = True,
            type             = "valid",
            trainpercentage  = 1.0,
            parrange         = self.parrange,
            label_i          = self.label_i_p2,
            extinction_mode  = "none",
            fixed_av         = 0.0,
            fixed_rv         = 3.1,
            split_seed       = self.split_seed,
            split            = split_dict,
        )

        # Sub-sample to n_test for manageable evaluation
        n_avail = len(ds_test)
        n_use   = min(self.n_test, n_avail)
        rng2    = np.random.default_rng(self.seed)
        idx     = rng2.choice(n_avail, size=n_use, replace=False)
        idx     = np.sort(idx)

        if self.verbose:
            print(f"[Diagnostics] Test set: {n_avail} available, "
                  f"using {n_use}")

        # Collect spectra and parameters
        loader = torch.utils.data.DataLoader(
            XYFromFlat(ds_test),
            batch_size  = self.batch_size,
            sampler     = torch.utils.data.SubsetRandomSampler(idx),
            num_workers = 0,
            drop_last   = False,
        )

        x_list, y_list = [], []
        with torch.no_grad():
            for xb, yb in loader:
                x_list.append(xb.cpu())
                y_list.append(yb.cpu())

        self.x_test = torch.cat(x_list, dim=0)   # (N, d_full)
        self.y_test = torch.cat(y_list, dim=0)   # (N, L)  raw flux

        # Parameter arrays for plotting
        d_phys = self.model.d_phys
        self.params_test = {
            lab: self.x_test[:, i].numpy()
            for i, lab in enumerate(self.label_i_p2[:d_phys])
        }
        if len(self.label_i_p2) > d_phys:
            self.params_test["av"] = self.x_test[:, d_phys].numpy()
            self.params_test["rv"] = self.x_test[:, d_phys + 1].numpy()

        # Run inference once and cache results
        self._run_inference()

    def _run_inference(self):
        """Run forward pass on the full test set and cache residuals."""
        self.model.eval()
        yhat_list, khat_list = [], []

        with torch.no_grad():
            for i in range(0, self.x_test.size(0), self.batch_size):
                xb        = self.x_test[i: i + self.batch_size].to(device)
                yh, kh    = self.model(xb, return_khat=True)
                yhat_list.append(yh.cpu())
                khat_list.append(kh.cpu())

        self.yhat_log = torch.cat(yhat_list, dim=0).numpy()   # (N, L) predicted log10
        self.khat     = torch.cat(khat_list, dim=0).numpy()   # (N, L) extinction curve
        self.ytrue_log = np.log10(
            np.clip(self.y_test.numpy(), 1e-12, None)
        )                                                       # (N, L) true log10

        # Residuals in dex (predicted − true)
        self.resid_dex = self.yhat_log - self.ytrue_log        # (N, L)

        if self.verbose:
            rms_global = float(np.sqrt(np.mean(self.resid_dex ** 2)))
            med_abs    = float(np.median(np.abs(self.resid_dex)))
            print(f"[Diagnostics] Global RMS residual : {rms_global:.5f} dex")
            print(f"[Diagnostics] Global median |Δ|   : {med_abs:.5f} dex")

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------
    def _savefig(self, fig: plt.Figure, name: str):
        path = os.path.join(self.plotdir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if self.verbose:
            print(f"  Saved: {path}")

    @staticmethod
    def _percentile_band(arr: np.ndarray, axis: int = 0):
        """Return (p16, p50, p84) along *axis*."""
        return (np.percentile(arr, 16, axis=axis),
                np.percentile(arr, 50, axis=axis),
                np.percentile(arr, 84, axis=axis))

    # ================================================================
    # Plot 1 — spectral residuals
    # ================================================================
    def plot_spectral_residuals(self, n_examples: int = 5):
        """True vs predicted spectra and residual percentile envelope.

        Top panel   : example predicted (solid) and true (dashed) log10 flux
        Middle panel: residual envelope (16/50/84th percentile across test set)
        Bottom panel: |residual| percentile envelope in dex
        """
        wav = self.wavelengths_A

        p16, p50, p84 = self._percentile_band(self.resid_dex, axis=0)
        ap16, ap50, ap84 = self._percentile_band(np.abs(self.resid_dex), axis=0)

        rng   = np.random.default_rng(self.seed)
        ex_idx = rng.choice(self.yhat_log.shape[0], size=n_examples, replace=False)
        cmap   = plt.cm.viridis(np.linspace(0.15, 0.85, n_examples))

        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(3, 1, figsize=(10, 9),
                                     sharex=True, layout="constrained")

            # -- top: example spectra --
            ax = axes[0]
            for ii, (ei, col) in enumerate(zip(ex_idx, cmap)):
                ax.plot(wav, self.ytrue_log[ei],  color=col, lw=0.8,
                        ls="--", alpha=0.8)
                ax.plot(wav, self.yhat_log[ei],   color=col, lw=0.8,
                        alpha=0.9,
                        label=(f"star {ii+1}" if ii == 0 else None))
            ax.set_ylabel(r"$\log_{10}$ flux")
            ax.set_title("Example spectra: predicted (solid) vs true (dashed)")
            # custom legend
            handles = [
                Line2D([0],[0], color="k", ls="-",  lw=1.5, label="predicted"),
                Line2D([0],[0], color="k", ls="--", lw=1.5, label="true"),
            ]
            ax.legend(handles=handles, loc="best")

            # -- middle: signed residual envelope --
            ax = axes[1]
            ax.fill_between(wav, p16, p84, alpha=0.25, color="C0",
                            label="16–84th pct")
            ax.plot(wav, p50, color="C0", lw=1.0, label="median")
            ax.axhline(0, color="k", lw=0.6, ls="--")
            ax.set_ylabel(r"$\hat{y} - y_{\rm true}$ (dex)")
            ax.set_title("Signed residual distribution across test set")
            ax.legend(loc="best")

            # -- bottom: |residual| envelope --
            ax = axes[2]
            ax.fill_between(wav, ap16, ap84, alpha=0.25, color="C3",
                            label="16–84th pct")
            ax.plot(wav, ap50, color="C3", lw=1.0, label="median")
            ax.set_ylabel(r"$|\hat{y} - y_{\rm true}|$ (dex)")
            ax.set_xlabel(r"Wavelength ($\AA$)")
            ax.set_title("Absolute residual distribution across test set")
            ax.legend(loc="best")

            for ax in axes:
                ax.set_xlim(wav[0], wav[-1])

        self._savefig(fig, self.outputtag + "spectral_residuals.png")

    # ================================================================
    # Plot 2 — residuals vs stellar parameters
    # ================================================================
    def plot_residuals_vs_params(self, n_bins: int = 15):
        """Median absolute residual (dex) vs each stellar parameter.

        Each panel shows the per-star median |Δ| as a scatter point,
        overlaid with a running median binned across the parameter axis.
        """
        phys_labels = [l for l in self.label_i_p2
                       if l not in ("av", "rv")]
        n_panels = len(phys_labels)

        # Per-star median absolute residual
        star_mad = np.median(np.abs(self.resid_dex), axis=1)   # (N,)

        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, n_panels,
                                     figsize=(4.5 * n_panels, 4.5),
                                     layout="constrained")
            if n_panels == 1:
                axes = [axes]

            for ax, lab in zip(axes, phys_labels):
                par = self.params_test[lab]
                ax.scatter(par, star_mad, s=6, alpha=0.35, color="C0",
                           rasterized=True)

                # binned median
                edges  = np.percentile(par,
                                       np.linspace(0, 100, n_bins + 1))
                edges  = np.unique(edges)
                bx, by = [], []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    mask = (par >= lo) & (par <= hi)
                    if mask.sum() > 3:
                        bx.append(0.5 * (lo + hi))
                        by.append(np.median(star_mad[mask]))
                ax.plot(bx, by, color="C3", lw=2.0, zorder=5)

                ax.set_xlabel(PARAM_LABELS.get(lab, lab))
                ax.set_ylabel(r"median $|\Delta|$ (dex)" if lab == phys_labels[0]
                              else "")
                ax.set_title(PARAM_LABELS.get(lab, lab))
                ax.set_ylim(bottom=0)

        self._savefig(fig, self.outputtag + "residuals_vs_params.png")

    # ================================================================
    # Plot 3 — PCA coefficient accuracy
    # ================================================================
    def plot_pca_coefficients(self, n_modes: int = 12):
        """Scatter z_true vs z_hat for the first n_modes PCA coefficients.

        Only meaningful when the model was trained with a PCA basis.
        """
        if not self.model.has_basis:
            if self.verbose:
                print("[Diagnostics] No PCA basis — skipping pca_coefficients plot.")
            return

        # Project true and predicted log spectra onto the PCA basis
        mu  = (self.model.mu_log.cpu().numpy()
               if self.model.mu_log is not None else 0.0)
        B   = self.model.f0.B.B.cpu().numpy()   # (K, L)
        K   = B.shape[0]

        z_true = (self.ytrue_log - mu) @ B.T    # (N, K)
        z_hat  = (self.yhat_log  - mu) @ B.T    # (N, K)

        n_show = min(n_modes, K)
        ncols  = 4
        nrows  = int(np.ceil(n_show / ncols))

        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(4 * ncols, 3.5 * nrows),
                                     layout="constrained")
            axes_flat = np.array(axes).flatten()

            for mi in range(n_show):
                ax  = axes_flat[mi]
                zt  = z_true[:, mi]
                zh  = z_hat[:, mi]
                rms = float(np.sqrt(np.mean((zh - zt) ** 2)))
                r2  = float(1.0 - np.var(zh - zt) / np.var(zt))

                ax.scatter(zt, zh, s=4, alpha=0.25, color="C0", rasterized=True)
                lo = min(zt.min(), zh.min())
                hi = max(zt.max(), zh.max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
                ax.set_title(f"Mode {mi+1}  RMS={rms:.3f}  R²={r2:.4f}",
                             fontsize=9)
                ax.set_xlabel(r"$z_{\rm true}$", fontsize=9)
                ax.set_ylabel(r"$\hat{z}$",      fontsize=9)

            # Hide unused panels
            for mi in range(n_show, len(axes_flat)):
                axes_flat[mi].set_visible(False)

        self._savefig(fig, self.outputtag + "pca_coefficients.png")

    # ================================================================
    # Plot 4 — Line profile accuracy
    # ================================================================
    def plot_line_profile_accuracy(
        self,
        line_centers_A: Tuple[float, ...] = (5889.95, 5895.92),
        window_A: float = 8.0,
        n_per_group: int = 5,
    ):
        """Zoom into two windows for representative stellar types. The
        default is the Na D doublet, but you can adjust line_centers_A and window_A as needed.

        Stars are grouped into four astrophysically motivated bins:
        hot/cool × metal-rich/metal-poor. For each group a few example
        spectra are overplotted showing true (dashed) vs predicted (solid).
        """
        wav    = self.wavelengths_A
        logt   = self.params_test.get("logt", None)
        feh    = self.params_test.get("feh",  None)

        if logt is None or feh is None:
            if self.verbose:
                print("[Diagnostics] logt/feh not found — skipping line profile plot.")
            return

        # Group boundaries (adjust to your grid)
        t_mid  = np.median(logt)
        fe_mid = np.median(feh)

        groups = {
            "cool, metal-rich":  (logt <= t_mid) & (feh >= fe_mid),
            "cool, metal-poor":  (logt <= t_mid) & (feh <  fe_mid),
            "hot,  metal-rich":  (logt >  t_mid) & (feh >= fe_mid),
            "hot,  metal-poor":  (logt >  t_mid) & (feh <  fe_mid),
        }

        n_lines = len(line_centers_A)
        fig, axes = plt.subplots(
            len(groups), n_lines,
            figsize=(5 * n_lines, 3.5 * len(groups)),
            layout="constrained",
        )
        axes = np.atleast_2d(axes)

        rng = np.random.default_rng(self.seed + 10)

        for row, (grp_label, mask) in enumerate(groups.items()):
            idx_avail = np.where(mask)[0]
            if len(idx_avail) == 0:
                continue
            n_pick = min(n_per_group, len(idx_avail))
            chosen = rng.choice(idx_avail, size=n_pick, replace=False)
            cmap   = plt.cm.plasma(np.linspace(0.15, 0.85, n_pick))

            for col, lc in enumerate(line_centers_A):
                ax      = axes[row, col]
                wm      = (wav >= lc - window_A/2) & (wav <= lc + window_A/2)
                wav_sub = wav[wm]

                for ii, (ci, col_c) in enumerate(zip(chosen, cmap)):
                    yt = self.ytrue_log[ci, wm]
                    yh = self.yhat_log[ci, wm]
                    # Normalise to continuum for clean shape comparison
                    cont = np.percentile(yt, 90)
                    ax.plot(wav_sub, yt - cont, color=col_c, lw=0.9,
                            ls="--", alpha=0.8)
                    ax.plot(wav_sub, yh - cont, color=col_c, lw=0.9,
                            alpha=0.9)

                ax.axvline(lc, color="k", lw=0.5, ls=":")
                ax.set_xlabel(r"Wavelength ($\AA$)")
                ax.set_ylabel(r"Normalised $\log_{10}$ flux" if col == 0 else "")
                ax.set_title(
                    f"{grp_label}\n"
                    f"Na D {lc:.2f} Å"
                )
                ax.set_xlim(lc - window_A/2, lc + window_A/2)

        # Global legend
        handles = [
            Line2D([0],[0], color="k", ls="-",  lw=1.5, label="predicted"),
            Line2D([0],[0], color="k", ls="--", lw=1.5, label="true"),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=10)

        self._savefig(fig, self.outputtag + "line_profile_accuracy.png")

    # ================================================================
    # Plot 5 — learned extinction curve vs G23
    # ================================================================
    def plot_khat_extinction_curve(
        self,
        rv_values: Tuple[float, ...] = (2.3, 3.1, 4.0, 5.6),
        n_stars: int = 200,
    ):
        """Compare the model's learned k̂(λ) to the true G23 extinction curve.

        k̂(λ) is evaluated by running the khat head for a random sample of
        test stars at each R_V value (with Av=1 for interpretability), then
        averaging over stellar parameters to show the mean curve ± 1σ.
        The G23 reference is overplotted as a dashed line.
        """
        wav  = self.wavelengths_A
        cmap = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(rv_values)))

        # Stellar-only inputs for a random subset
        rng     = np.random.default_rng(self.seed + 99)
        n_pick  = min(n_stars, self.x_test.size(0))
        idx_sub = rng.choice(self.x_test.size(0), size=n_pick, replace=False)
        x_sub   = self.x_test[idx_sub]                        # (n_pick, d_full)

        d_phys  = self.model.d_phys

        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

            for rv_val, col in zip(rv_values, cmap):
                # Build input tensor with fixed Av=1, varying Rv
                x_rv      = x_sub.clone()
                x_rv[:, d_phys]     = 1.0   # Av = 1 (unnormalised for khat)
                x_rv[:, d_phys + 1] = rv_val

                x_phys_t = x_rv[:, :d_phys].to(device)
                rv_t      = x_rv[:, d_phys + 1:d_phys + 2].to(device)

                self.model.eval()
                with torch.no_grad():
                    inp      = torch.cat([x_phys_t, rv_t], dim=1)
                    k_hat_rv = self.model.khat(inp).cpu().numpy()  # (n_pick, L)

                k_mean = k_hat_rv.mean(axis=0)
                k_std  = k_hat_rv.std(axis=0)

                ax.plot(wav, k_mean, color=col, lw=1.8,
                        label=f"$R_V$={rv_val:.1f}")
                ax.fill_between(wav, k_mean - k_std, k_mean + k_std,
                                color=col, alpha=0.15)

                # G23 reference
                lo_rv, hi_rv = 2.3, 5.6
                rv_clamped   = float(np.clip(
                    rv_val,
                    np.nextafter(lo_rv, 10.0),
                    np.nextafter(hi_rv, 0.0),
                ))
                x_inv  = (1.0 / (wav * 1e-4)) * u.micron ** -1
                k_true = np.array(G23(Rv=rv_clamped)(x_inv), dtype=np.float64)
                ax.plot(wav, k_true, color=col, lw=1.0, ls="--", alpha=0.7)

            ax.set_xlabel(r"Wavelength ($\AA$)")
            ax.set_ylabel(r"$\hat{k}(\lambda) = A(\lambda)/A_V$")
            ax.set_title(r"Learned extinction curve $\hat{k}(\lambda)$ "
                         r"(solid ± 1σ) vs G23 reference (dashed)")
            ax.set_xlim(wav[0], wav[-1])
            ax.set_ylim(bottom=0)

            # Add Rv colourbar-style legend
            handles = [
                Line2D([0],[0], color=col, lw=2.0, label=f"$R_V$={rv:.1f}")
                for rv, col in zip(rv_values, cmap)
            ]
            handles += [
                Line2D([0],[0], color="k", ls="--", lw=1.2, label="G23 (true)")
            ]
            ax.legend(handles=handles, loc="best")

        self._savefig(fig, self.outputtag + "khat_extinction_curve.png")

    # ================================================================
    # Plot 6 — HR diagram coloured by residual
    # ================================================================
    def plot_hr_diagram_residuals(self):
        """Kiel diagram (logt vs logg) coloured by per-star RMS residual.

        Reveals which regions of the HR diagram the emulator finds hardest.
        """
        logt = self.params_test.get("logt", None)
        logg = self.params_test.get("logg", None)
        if logt is None or logg is None:
            if self.verbose:
                print("[Diagnostics] logt/logg not in params — skipping HR diagram.")
            return

        star_rms = np.sqrt(np.mean(self.resid_dex ** 2, axis=1))   # (N,)

        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                                     layout="constrained")

            # -- left: RMS residual --
            vmax = np.percentile(star_rms, 95)
            sc   = axes[0].scatter(
                logt, logg, c=star_rms, s=8, alpha=0.7,
                cmap="inferno_r", vmin=0, vmax=vmax, rasterized=True,
            )
            axes[0].invert_xaxis()
            axes[0].invert_yaxis()
            axes[0].set_xlabel(PARAM_LABELS["logt"])
            axes[0].set_ylabel(PARAM_LABELS["logg"])
            axes[0].set_title("RMS residual (dex)")
            fig.colorbar(sc, ax=axes[0], label="RMS (dex)")

            # -- right: signed median residual (bias check) --
            star_bias = np.median(self.resid_dex, axis=1)   # (N,)
            vlim = np.percentile(np.abs(star_bias), 95)
            sc2  = axes[1].scatter(
                logt, logg, c=star_bias, s=8, alpha=0.7,
                cmap="RdBu_r", vmin=-vlim, vmax=vlim, rasterized=True,
            )
            axes[1].invert_xaxis()
            axes[1].invert_yaxis()
            axes[1].set_xlabel(PARAM_LABELS["logt"])
            axes[1].set_ylabel(PARAM_LABELS["logg"])
            axes[1].set_title("Median signed residual (dex) — bias map")
            fig.colorbar(sc2, ax=axes[1], label="median Δ (dex)")

        self._savefig(fig, self.outputtag + "hr_diagram_residuals.png")

    # ================================================================
    # Run all plots
    # ================================================================
    def run_all(
        self,
        n_examples:    int = 5,
        n_modes:       int = 12,
        n_per_group:   int = 5,
        rv_values:     Tuple[float, ...] = (2.3, 3.1, 4.0, 5.6),
        line_centers_A: Tuple[float, ...] = (5889.95, 5895.92),
        window_A:      float = 8.0,
    ):
        """Generate all diagnostic plots in sequence."""
        print("\n[Diagnostics] Generating diagnostic plots ...")
        print(f"  Output directory: {self.plotdir}")

        print("  1/6  spectral residuals ...")
        self.plot_spectral_residuals(n_examples=n_examples)

        print("  2/6  residuals vs parameters ...")
        self.plot_residuals_vs_params()

        print("  3/6  PCA coefficient accuracy ...")
        self.plot_pca_coefficients(n_modes=n_modes)

        print("  4/6  line profile accuracy ...")
        self.plot_line_profile_accuracy(n_per_group=n_per_group, line_centers_A=line_centers_A, window_A=window_A)

        print("  5/6  learned extinction curve ...")
        self.plot_khat_extinction_curve(rv_values=rv_values)

        print("  6/6  HR diagram residuals ...")
        self.plot_hr_diagram_residuals()

        print("[Diagnostics] All plots saved.")