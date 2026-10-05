"""readphot_bands

ReadPhotBands: photometric training data with extinction integrated through each
filter, from the per-star band weights that cwc/phot genphot writes to
/bandweights/<system> (genphot --band-weights).

For star s, filter f and extinction (Av, Rv):

    BC_f(Av, Rv) = BC_f,0 + 2.5 log10 sum_b w_fb 10^(-0.4 Av k(lam_fb; Rv))

where BC_f,0 is the Av=0 BC in /<system>, w_fb the fraction of the Av=0 detector
counts in extinction bin b and lam_fb the counts-weighted mean wavelength of the
bin. ReadPhot instead uses BC_f,0 - Av k(lambda_pivot), which is wrong by tenths of
a magnitude or more for broad bands (Gaia) and the blue SPHEREx channels.

ReadPhotBands subclasses ReadPhot, so the split, parrange, normalisation, filter
mapping, Av/Rv grids and extinction modes are ReadPhot's own; only the targets
change. It adds a device-side batch API for trainers:

    ds.to_device(device)
    x, y = ds.batch(rows, av, rv)             # rows index ds.rows (unique stars)
    av, rv = ds.sample_extinction(n, gen)     # ReadPhot 'sample' mode, on device
    rows, av, rv = ds.grid_items()            # every item of a 'grid'/'fixed' dataset
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch

from dust_extinction.parameter_averages import G23
from astropy import units as u

from .readKorg_hybrid_ext import ReadPhot, _boogert_k_av, _normalise_extinction_law

__all__ = ["ReadPhotBands", "ExtinctionTable"]

LN10 = math.log(10.0)


class ExtinctionTable:
    """
    A(lambda)/A(V) as  k(lambda, Rv) = a(lambda) + b(lambda) * (1/Rv - 1/3.1),
    tabulated on a uniform log-lambda grid and linearly interpolated (NumPy or
    torch). G23 has exactly this Rv dependence; Boogert+2011 has b = 0. For the
    'hybrid' law, samples with Av < av_break use G23 and the rest Boogert.
    """

    RV_REF = 3.1
    LAM_MIN, LAM_MAX = 913.0, 3.19e5  # Angstrom; G23 domain is 0.0912-32 micron

    def __init__(self, law: str = "g23", av_break: float = 2.0,
                 boogert_av_to_ak: float = 1.0 / 7.045, npts: int = 20000):
        self.law = _normalise_extinction_law(law)
        self.av_break = float(av_break)
        self.loglam0 = math.log(self.LAM_MIN)
        self.dloglam = (math.log(self.LAM_MAX) - self.loglam0) / (npts - 1)
        lam = np.exp(self.loglam0 + self.dloglam * np.arange(npts))
        lam_um = lam * 1e-4 * u.micron
        k31 = np.asarray(G23(Rv=self.RV_REF)(lam_um), dtype=np.float64)
        k23 = np.asarray(G23(Rv=2.3)(lam_um), dtype=np.float64)
        self.g23_a = k31
        self.g23_b = (k23 - k31) / (1.0 / 2.3 - 1.0 / self.RV_REF)
        self.boogert = np.asarray(_boogert_k_av(lam * 1e-4, boogert_av_to_ak), dtype=np.float64)
        self._torch: Dict[str, torch.Tensor] = {}

    # ---- interpolation weights on the log-lambda grid ----
    def _pos_np(self, lam):
        x = (np.log(np.clip(lam, self.LAM_MIN, self.LAM_MAX)) - self.loglam0) / self.dloglam
        i = np.clip(np.floor(x).astype(np.int64), 0, self.g23_a.size - 2)
        return i, x - i

    def k_np(self, lam: np.ndarray, rv: float, av: float = 0.0) -> np.ndarray:
        i, f = self._pos_np(lam)
        lerp = lambda t: t[i] * (1.0 - f) + t[i + 1] * f
        if self.law == "boogert" or (self.law == "hybrid" and av >= self.av_break):
            return lerp(self.boogert)
        return lerp(self.g23_a) + lerp(self.g23_b) * (1.0 / rv - 1.0 / self.RV_REF)

    def to(self, device):
        self._torch = {k: torch.as_tensor(getattr(self, k), dtype=torch.float32, device=device)
                       for k in ("g23_a", "g23_b", "boogert")}
        return self

    def k_torch(self, lam: torch.Tensor, rv: torch.Tensor, av: torch.Tensor) -> torch.Tensor:
        """lam (B, NB), rv and av (B,) -> k (B, NB)."""
        T = self._torch
        x = (torch.log(lam.clamp(self.LAM_MIN, self.LAM_MAX)) - self.loglam0) / self.dloglam
        i = x.floor().long().clamp_(0, T["g23_a"].numel() - 2)
        f = x - i
        lerp = lambda t: t[i] * (1.0 - f) + t[i + 1] * f
        if self.law == "boogert":
            return lerp(T["boogert"])
        k = lerp(T["g23_a"]) + lerp(T["g23_b"]) * (1.0 / rv - 1.0 / self.RV_REF)[:, None]
        if self.law == "hybrid":
            k = torch.where((av >= self.av_break)[:, None], lerp(T["boogert"]), k)
        return k


class ReadPhotBands(ReadPhot):
    """ReadPhot with band-integrated extinction from /bandweights (see module docstring)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if "__single__" not in self.modpaths:
            raise NotImplementedError("ReadPhotBands reads one BC file (modpath=<file>).")
        path = self.modpaths["__single__"]

        # Unique stars used by this dataset (model_index = row in the HDF5)
        self.rows = np.unique(np.asarray(self._selind, dtype=np.int64))
        self._item_row = np.searchsorted(self.rows, np.asarray(self._selind, dtype=np.int64))

        # Gather this dataset's columns in label_o order, filter by filter
        w_parts, lam_parts, seg_parts, bc0 = [], [], [], []
        with h5py.File(path, "r") as h5:
            if "bandweights" not in h5:
                raise KeyError(f"{path} has no /bandweights; rebuild it with genphot/assemblegrid --band-weights")
            cache = {}
            for j, (system, band, _) in enumerate(self._filter_map):
                if system not in cache:
                    g = h5[f"bandweights/{system}"]
                    names = [n.decode() if isinstance(n, bytes) else str(n) for n in g["filters"][()]]
                    cache[system] = dict(
                        names={n.lower(): i for i, n in enumerate(names)},
                        offsets=g["offsets"][()],
                        w=g["w"][self.rows, :],
                        lam=g["lam"][self.rows, :],
                        dk=float(g.attrs["dk"]),
                    )
                c = cache[system]
                fi = c["names"][band.lower()]
                cols = slice(int(c["offsets"][fi]), int(c["offsets"][fi + 1]))
                w_parts.append(c["w"][:, cols])
                lam_parts.append(c["lam"][:, cols])
                seg_parts.append(np.full(cols.stop - cols.start, j, dtype=np.int64))
                bc0.append(np.asarray(self.h5dict[system][band][self.rows], dtype=np.float64))
        self.band_w = np.concatenate(w_parts, axis=1).astype(np.float32)
        self.band_lam = np.concatenate(lam_parts, axis=1).astype(np.float32)
        self.band_seg = np.concatenate(seg_parts)
        self.band_starts = np.concatenate([[0], np.cumsum([p.shape[1] for p in w_parts])[:-1]])
        self.bc0 = np.stack(bc0, axis=1)  # (n_rows, n_out), physical BC at Av=0
        self.band_dk = {s: c["dk"] for s, c in cache.items()}
        if np.isnan(self.band_w).any() or np.isnan(self.bc0).any():
            raise ValueError(f"{path}: NaN band weights or BCs for some selected rows (rows never written?)")

        self.ext = ExtinctionTable(self.extinction_law, self.extinction_av_break, self.boogert_av_to_ak)
        self._dev: Dict[str, torch.Tensor] = {}
        if self.verbose:
            print(f"[ReadPhotBands] {len(self.rows)} stars, {self.band_w.shape[1]} extinction bins "
                  f"over {len(self.label_o)} outputs (dk={self.band_dk}), law={self.extinction_law}")

    # ------------------------------------------------------------------
    # NumPy path (Dataset API, used by plotting / diagnostics)
    # ------------------------------------------------------------------
    def band_bc(self, row: int, av: float, rv: float) -> np.ndarray:
        """Physical BCs (n_out,) of local star `row` at (Av, Rv)."""
        if av == 0.0:
            return self.bc0[row].copy()
        with np.errstate(divide="ignore"):
            logs = np.log(self.band_w[row].astype(np.float64)) \
                - 0.4 * LN10 * av * self.ext.k_np(self.band_lam[row].astype(np.float64), rv, av)
        m = np.maximum.reduceat(logs, self.band_starts)
        s = np.add.reduceat(np.exp(logs - m[self.band_seg]), self.band_starts)
        return self.bc0[row] + (2.5 / LN10) * (m + np.log(s))

    def __getitem__(self, idx: int):
        row = self._param_rows[idx]
        if self.extinction_mode == "grid":
            gpos = idx % self._per_row_grid
            av, rv = float(self._grid_av[gpos]), float(self._grid_rv[gpos])
        elif self.extinction_mode == "fixed":
            av, rv = self.fixed_av, self.fixed_rv
        elif self.extinction_mode == "sample":
            av, rv = float(self.rng.choice(self.avgrid)), float(self.rng.choice(self.rvgrid))
        else:
            av, rv = 0.0, 3.1

        bc = self.band_bc(int(self._item_row[idx]), av if self.extinction_mode != "none" else 0.0, rv)
        bcout = [self.normf(b, lab) if self.norm else b for b, lab in zip(bc, self.label_o)]

        inputs: List[float] = []
        for ll in self.label_i:
            if ll in row.dtype.names:
                val = float(row[ll])
            elif ll == "av":
                val = av
            elif ll == "rv":
                val = rv
            else:
                raise KeyError(f"Input label '{ll}' not found in parameters or av/rv.")
            inputs.append(self.normf(val, ll) if self.norm else val)

        flat = np.array(inputs + bcout, dtype=np.float32)
        return torch.tensor(flat) if self.returntorch else flat

    # ------------------------------------------------------------------
    # Device path (trainers)
    # ------------------------------------------------------------------
    def to_device(self, device):
        """Move everything batch() needs to `device`; returns self."""
        dev = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
        with np.errstate(divide="ignore"):
            logw = np.log(self.band_w)
        # per-star physical stellar labels in label_i order (av/rv filled per batch)
        # (ReadPhot may have shuffled self.parameters in place, so look rows up via argsort)
        mi = self.parameters["model_index"]
        order = np.argsort(mi)
        params = self.parameters[order[np.searchsorted(mi[order], self.rows)]]
        assert np.array_equal(params["model_index"], self.rows)
        lab_cols = np.zeros((len(self.rows), len(self.label_i)), dtype=np.float64)
        for i, ll in enumerate(self.label_i):
            if ll in params.dtype.names:
                lab_cols[:, i] = params[ll]
        nf = lambda labs: (dev([self.normfactor[l][0] for l in labs]), dev([self.normfactor[l][1] for l in labs]))
        self._dev = dict(
            logw=dev(logw), lam=dev(self.band_lam), seg=dev(self.band_seg, torch.long), bc0=dev(self.bc0),
            labels=dev(lab_cols), in_norm=nf(self.label_i), out_norm=nf(self.label_o),
            avgrid=dev(self.avgrid), rvgrid=dev(self.rvgrid),
        )
        self._av_col = self.label_i.index("av") if "av" in self.label_i else None
        self._rv_col = self.label_i.index("rv") if "rv" in self.label_i else None
        self.ext.to(device)
        self.device = torch.device(device)
        return self

    def batch(self, rows: torch.Tensor, av: torch.Tensor, rv: torch.Tensor, norm: Optional[bool] = None):
        """
        rows: LongTensor of local star indices (into self.rows); av, rv: (B,) tensors.
        Returns (x, y), normalised like ReadPhot when norm (default self.norm).
        """
        D = self._dev
        norm = self.norm if norm is None else norm
        av = av.to(torch.float32); rv = rv.to(torch.float32)
        logs = D["logw"][rows] - (0.4 * LN10) * av[:, None] * self.ext.k_torch(D["lam"][rows], rv, av)
        B, nout = logs.shape[0], D["bc0"].shape[1]
        seg = D["seg"].expand(B, -1)
        m = torch.full((B, nout), -torch.inf, device=logs.device).scatter_reduce(1, seg, logs, "amax", include_self=True)
        s = torch.zeros((B, nout), device=logs.device).scatter_add_(1, seg, torch.exp(logs - m.gather(1, seg)))
        y = D["bc0"][rows] + (2.5 / LN10) * (m + torch.log(s))

        x = D["labels"][rows].clone()
        if self._av_col is not None:
            x[:, self._av_col] = av
        if self._rv_col is not None:
            x[:, self._rv_col] = rv
        if norm:
            x = (x - D["in_norm"][0]) / D["in_norm"][1]
            y = (y - D["out_norm"][0]) / D["out_norm"][1]
        return x, y

    def sample_extinction(self, n: int, generator: Optional[torch.Generator] = None):
        """Av, Rv drawn uniformly from avgrid and rvgrid (ReadPhot 'sample' mode), on device."""
        D = self._dev
        ia = torch.randint(len(D["avgrid"]), (n,), device=D["avgrid"].device, generator=generator)
        ir = torch.randint(len(D["rvgrid"]), (n,), device=D["rvgrid"].device, generator=generator)
        return D["avgrid"][ia], D["rvgrid"][ir]

    def grid_items(self):
        """(rows, av, rv) device tensors for every item, in Dataset order (grid/fixed/none modes)."""
        if self.extinction_mode == "sample":
            raise ValueError("grid_items() needs a deterministic extinction_mode ('grid', 'fixed' or 'none')")
        rows = torch.as_tensor(self._item_row, dtype=torch.long, device=self.device)
        n = len(self._item_row)
        if self.extinction_mode == "grid":
            g = np.arange(n) % self._per_row_grid
            av, rv = self._grid_av[g], self._grid_rv[g]
        elif self.extinction_mode == "fixed":
            av, rv = np.full(n, self.fixed_av), np.full(n, self.fixed_rv)
        else:
            av, rv = np.zeros(n), np.full(n, 3.1)
        return rows, torch.as_tensor(av, dtype=torch.float32, device=self.device), \
            torch.as_tensor(rv, dtype=torch.float32, device=self.device)
