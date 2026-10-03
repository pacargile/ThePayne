"""readKorg

Production-ready dataloader for synthetic spectral grids produced from Korg,
with on-the-fly dust extinction, flexible wavelength selection / resampling,
and deterministic train/valid/test splitting.

Also contains ReadPhot for bolometric-correction (BC) tables from Korg/GenPhot
with SEDpy-driven filter wavelength handling.

Key features
------------
- Supports a single HDF5, a directory of HDF5s, or a list of paths.
- Robust mapping from SEDpy filter names to HDF5 (system, band) fields via
  prefix rules and optional user aliases (ReadPhot).
- On-the-fly G23, Boogert+2011, or hybrid extinction with safe R_V
  clamping and a thread-safe per-worker RNG for 'sample' mode.
- Deterministic train/valid/test splitting or externally supplied splits.
- Optional z-score normalisation for inputs and outputs.
- Auto-detection of fixed-vmic grids: if all rows share the same vmic value,
  'vmic' is silently dropped from label_i so the network does not receive a
  constant feature.

Public API
----------
- ReadPhot  : Dataset yielding a flat vector [x_in || y_out] for photometry.
- ReadSpec  : Dataset yielding a flat vector [x_in || y_out] for spectra.
- XYFromFlat: Thin wrapper that splits the flat vector into (x, y) tensors.

Notes
-----
- Extinction modes: 'sample' (train default), 'grid' (valid/test default),
  'fixed', 'none'.
- 'sample' mode uses a per-worker numpy Generator seeded from
  (split_seed, worker_id) so that multi-process DataLoaders are safe.
- The k(lambda) cache keys are rounded to 4 d.p. to avoid float-equality
  misses caused by tiny floating-point rounding differences.
"""
from __future__ import annotations

import os
import glob
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from tqdm import tqdm

import h5py
import numpy as np
from numpy.lib import recfunctions as rfn

import torch
from torch.utils.data import Dataset

from dust_extinction.parameter_averages import G23
from astropy import units as u


# -----------------------------------------------------------------------
# Extinction-law helpers
# -----------------------------------------------------------------------
_BOOGERT_COEFFS = np.array(
    [0.5924, -1.8235, -1.3020, 5.9936, -5.3429,
     1.2619, 0.2738, 0.0069, -0.0554],
    dtype=np.float64,
)


def _normalise_extinction_law(name: str) -> str:
    """Return canonical extinction-law name."""
    law = str(name).strip().lower().replace("-", "_")
    aliases = {
        "g23": "g23",
        "gordon23": "g23",
        "gordon2023": "g23",
        "gordon_2023": "g23",
        "boogert": "boogert",
        "boogert11": "boogert",
        "boogert2011": "boogert",
        "boogert_2011": "boogert",
        "hybrid": "hybrid",
        "g23_boogert": "hybrid",
        "gordon23_boogert": "hybrid",
        "gordon2023_boogert2011": "hybrid",
    }
    if law not in aliases:
        raise ValueError(
            "extinction_law must be one of 'g23', 'boogert', or 'hybrid' "
            f"(got {name!r})."
        )
    return aliases[law]


def _active_extinction_law(extinction_law: str, av: float, av_break: float) -> str:
    """Resolve the actual law used for one A_V value."""
    law = _normalise_extinction_law(extinction_law)
    if law == "hybrid":
        return "g23" if float(av) < float(av_break) else "boogert"
    return law


def _boogert_k_av(wavelength_micron: np.ndarray | float, av_to_ak: float) -> np.ndarray:
    """Boogert+2011 A(lambda)/A(V) at wavelength in micron.

    The polynomial is normalised as A(lambda)/A(K). This helper keeps the
    dataloader extinction amplitude as A_V by multiplying by A_K/A_V.
    The default used below is 1/7.045, matching the supplied
    boogert_extinction.py example.
    """
    lam = np.asarray(wavelength_micron, dtype=np.float64)
    if np.any(lam <= 0):
        raise ValueError("Boogert extinction requires positive wavelengths in micron.")
    log_lam = np.log10(lam)
    log_a_over_ak = np.polynomial.polynomial.polyval(log_lam, _BOOGERT_COEFFS)
    return float(av_to_ak) * np.power(10.0, log_a_over_ak)


def _build_extinction_grid_pairs(
    avgrid: np.ndarray,
    rvgrid: np.ndarray,
    extinction_law: str,
    av_break: float,
    fixed_rv: float,
    collapse_hybrid_rv: bool,
) -> List[Tuple[float, float]]:
    """Return (A_V, R_V) grid pairs, optionally collapsing inactive R_V."""
    if extinction_law == "hybrid" and collapse_hybrid_rv:
        pairs: List[Tuple[float, float]] = []
        for avv in avgrid:
            if float(avv) < av_break:
                pairs.extend((float(avv), float(rvv)) for rvv in rvgrid)
            else:
                # Boogert+2011 has no R_V dependence.  Use one canonical
                # R_V input instead of duplicating identical outputs for every
                # R_V grid point.
                pairs.append((float(avv), float(fixed_rv)))
        return pairs
    return [(float(avv), float(rvv)) for avv in avgrid for rvv in rvgrid]


__all__ = ["ReadPhot", "ReadSpec", "XYFromFlat"]

# -----------------------------------------------------------------------
# Optional dependency: sedpy
# -----------------------------------------------------------------------
try:
    from sedpy import observate
    from sedpy.observate import Filter as SEDpyFilter
except Exception as e:   # pragma: no cover
    raise ImportError("`sedpy` is required when using `filters`.") from e


# -----------------------------------------------------------------------
# SEDpy wavelength helpers
# -----------------------------------------------------------------------
def _pivot_wavelength(wave_A: np.ndarray, trans: np.ndarray) -> float:
    """Pivot wavelength in Angstrom: sqrt(∫Sλ dλ / ∫S/λ dλ)."""
    w = np.asarray(wave_A, dtype=float)
    S = np.asarray(trans,  dtype=float)
    return float(np.sqrt(np.trapz(S * w, w) / np.trapz(S / w, w)))


def _logmean_wavelength(wave_A: np.ndarray, trans: np.ndarray) -> float:
    """Log-mean wavelength in Angstrom: exp(Σ ln λ · S · d ln λ / Σ S · d ln λ)."""
    w   = np.asarray(wave_A, dtype=float)
    S   = np.asarray(trans,  dtype=float)
    lnw = np.log(w)
    dlnw = np.gradient(lnw)
    return float(np.exp(np.sum(lnw * S * dlnw) / np.sum(S * dlnw)))


# -----------------------------------------------------------------------
# Filesystem utilities
# -----------------------------------------------------------------------
def _as_dict_of_paths(modpath: str | Mapping[str, str]) -> Dict[str, str]:
    """Normalise *modpath* to a {system: path} dict."""
    if isinstance(modpath, dict):
        for p in modpath.values():
            if not os.path.isfile(p):
                raise FileNotFoundError(f"HDF5 file not found: {p}")
        return dict(modpath)
    if os.path.isdir(modpath):
        out: Dict[str, str] = {}
        for p in sorted(glob.glob(os.path.join(modpath, "*.h5"))):
            out[os.path.splitext(os.path.basename(p))[0]] = p
        if not out:
            raise FileNotFoundError(f"No .h5 files under: {modpath}")
        return out
    if os.path.isfile(modpath):
        return {"__single__": modpath}
    raise FileNotFoundError(f"modpath not found: {modpath}")


def _first_attr(obj: object, names: Sequence[str]) -> Optional[np.ndarray]:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None and not callable(v):
                return v
    return None


def _get_filter_arrays(f: SEDpyFilter) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wave_A, trans) from a SEDpy Filter object."""
    wave  = _first_attr(f, ("wave", "wavelength", "lam", "_wavelength"))
    trans = _first_attr(f, ("trans", "throughput", "_transmission"))
    if wave is None or trans is None:
        raise TypeError(
            "Could not find wavelength/throughput on sedpy Filter. "
            "Tried wave|wavelength|lam|_wavelength and trans|throughput|_transmission."
        )
    return np.asarray(wave, dtype=float), np.asarray(trans, dtype=float)


# -----------------------------------------------------------------------
# SEDpy-name → (system, band) resolver
# -----------------------------------------------------------------------
_DEFAULT_PREFIX_MAP: Dict[str, Tuple[str, callable]] = {
    r"^ps_":      ("panstarrs", lambda n: n.split("_", 1)[1]),
    r"^gaia_":    ("gaia",      lambda n: n.split("_", 1)[1]),
    r"^twomass_": ("twomass",   lambda n: n.split("_", 1)[1]),
    r"^wise_":    ("wise",      lambda n: n.split("_", 1)[1]),
    r"^sdss_":    ("sdss",      lambda n: n.split("_", 1)[1].rstrip("0")),
    r"^decam_":   ("decam",     lambda n: n.split("_", 1)[1]),
    r"^lsst_":    ("lsst",      lambda n: n.split("_", 1)[1]),
    r"^roman_":   ("roman_wfi", lambda n: n.split("_", 2)[2]),
    r"^swift_":   ("uvot",      lambda n: n.split("_", 1)[1]),
    r"^spx_":     ("spherex",   lambda n: n.split("_", 1)[1]),
}


def _resolve_system_band_from_sedpy_name(
    sedpy_name: str,
    h5_system_names: Iterable[str],
    h5_fields_by_system: Mapping[str, Sequence[str]],
    user_system_alias: Optional[Mapping[str, str]] = None,
    user_band_alias: Optional[Mapping[Tuple[str, str], str]] = None,
) -> Tuple[str, str]:
    """Map a SEDpy filter name to (system, band) used in the HDF5."""
    name_lc = sedpy_name.lower()

    system: Optional[str] = None
    band_guess: Optional[str] = None
    for pref, (sysname, band_fn) in _DEFAULT_PREFIX_MAP.items():
        if re.match(pref, name_lc):
            system     = sysname
            band_guess = band_fn(sedpy_name)
            break
    if system is None or band_guess is None:
        parts      = sedpy_name.split("_", 1)
        system     = parts[0].lower() if len(parts) == 2 else sedpy_name.lower()
        band_guess = parts[1]         if len(parts) == 2 else sedpy_name

    if user_system_alias and system in user_system_alias:
        system = user_system_alias[system]
    if user_band_alias and (system, band_guess) in user_band_alias:
        band_guess = user_band_alias[(system, band_guess)]

    h5_system_names = list(h5_system_names)
    if system not in h5_system_names:
        cand = [s for s in h5_system_names
                if s.lower() == system.lower() or s.lower().startswith(system)]
        if len(cand) == 1:
            system = cand[0]
        elif len(cand) > 1:
            system = sorted(cand, key=len, reverse=True)[0]
        else:
            raise KeyError(
                f"System '{system}' (from '{sedpy_name}') not found in "
                f"HDF5 systems {sorted(h5_system_names)}"
            )

    fields = list(h5_fields_by_system.get(system, []))
    if not fields:
        raise KeyError(f"No filter fields found for system '{system}' in HDF5.")

    if band_guess in fields:
        return system, band_guess

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    fields_norm = {_norm(f): f for f in fields}
    band_norm   = _norm(band_guess)
    ci_map      = {f.lower(): f for f in fields}

    if band_guess.lower() in ci_map:
        return system, ci_map[band_guess.lower()]
    if band_norm in fields_norm:
        return system, fields_norm[band_norm]

    hits = [orig for norm, orig in fields_norm.items()
            if norm.startswith(band_norm) or band_norm.startswith(norm)]
    if len(hits) == 1:
        return system, hits[0]

    raise KeyError(
        f"Cannot map SEDpy filter '{sedpy_name}' to HDF5 ({system}, '{band_guess}'). "
        f"Available fields in '{system}': {sorted(fields)}"
    )


# -----------------------------------------------------------------------
# Worker-init helper for thread-safe sampling
# -----------------------------------------------------------------------
def _worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """Seed per-worker state so that 'sample' extinction mode is thread-safe.

    Call via::

        DataLoader(..., worker_init_fn=partial(_worker_init_fn, base_seed=seed))

    Each worker gets its own numpy Generator stored on the dataset object so
    concurrent workers never share RNG state.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    ds = worker_info.dataset
    # Walk through XYFromFlat wrapper if present
    base_ds = getattr(ds, "ds", ds)
    seed = (base_seed if base_seed is not None else 0) + worker_id
    base_ds.rng = np.random.default_rng(seed)


# -----------------------------------------------------------------------
# ReadPhot
# -----------------------------------------------------------------------
class ReadPhot(Dataset):
    """Dataset of synthetic photometry (BC tables) with optional G23 extinction.

    Parameters
    ----------
    modpath : str | dict
        HDF5 path (single-file or directory) or {system: path} mapping.
    filters : list[str]
        SEDpy filter names (e.g., ['ps_g', 'ps_r', 'gaia_g', ...]).
    filter_wavelength_method : {'pivot', 'logmean'}, default 'pivot'
        Method for per-filter representative wavelength.
    system_alias, band_alias : dict, optional
        Aliases for resolving (system, band) from SEDpy names.
    type : {'train', 'valid', 'test'}, default 'train'
    extinction_mode : {'sample', 'grid', 'fixed', 'none'}
        'sample' draws A_V and R_V randomly from avgrid/rvgrid each call.
        'grid' expands each stellar row by all (A_V, R_V) pairs.
        'fixed' applies a single (fixed_av, fixed_rv) to every row.
        'none' returns intrinsic magnitudes.
    avgrid, rvgrid : Sequence[float]
        Discrete grids for A_V and R_V sampling / expansion.
    fixed_av, fixed_rv : float
        Used when extinction_mode == 'fixed'.
    norm : bool, default True
        Apply z-score normalisation to inputs and outputs.
    normfactor : Mapping[str, Tuple[float, float]], optional
        Override normalisation stats (label → (mean, std)).
    split : dict[str, np.ndarray], optional
        External split indices keyed 'train' / 'valid' / 'test'.
    split_seed : int, optional
        RNG seed for deterministic auto-splitting.
    trainpercentage : float, default 0.9
        Fraction used for train+valid when auto-splitting.
    label_i : list[str]
        Input feature names. Default ['logt','logg','feh','afe','av','rv'].
    returntorch : bool, default True
    verbose : bool, default False
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.verbose: bool = kwargs.get("verbose", False)

        self.split_seed = kwargs.get("split_seed", kwargs.get("seed", None))
        self.rng        = np.random.default_rng(self.split_seed)
        self.split: Optional[Dict[str, np.ndarray]] = kwargs.get("split", None)
        self.normfactor_override = kwargs.get("normfactor", None)

        # ---- sources ----
        modpath = kwargs.get("modpath", None)
        if modpath is None:
            raise ValueError("Provide `modpath` (file, directory, or dict).")
        self.modpaths = _as_dict_of_paths(modpath)

        self.h5dict:   Dict[str, np.ndarray] = {}
        self._systems: List[str] = []

        sedpy_filters = kwargs.get("filters", None)
        if not sedpy_filters:
            raise ValueError("Pass `filters` (list of filter names).")

        # ---- load HDF5(s) ----
        if "__single__" in self.modpaths:
            path = self.modpaths["__single__"]
            with h5py.File(path, "r") as h5:
                if "parameters" not in h5:
                    raise KeyError("HDF5 must contain '/parameters'.")
                self.parameters = h5["parameters"][()]
                self.meta       = dict(h5["meta"].attrs) if "meta" in h5 else {}

                NON_SYS = {"parameters", "meta", "rowkey"}
                requested_systems = self._infer_systems_from_filters(sedpy_filters)
                available         = {k for k, v in h5.items()
                                     if isinstance(v, h5py.Dataset) and k not in NON_SYS}
                missing = sorted(requested_systems - available)
                if missing:
                    raise KeyError(
                        f"Requested systems {missing} not in HDF5. "
                        f"Available: {sorted(available)}"
                    )
                for sysname in sorted(requested_systems):
                    ds = h5[sysname]
                    if ds.dtype.names is None:
                        raise TypeError(f"/{sysname} must be a structured array.")
                    self.h5dict[sysname] = ds[()]
                    self._systems.append(sysname)
        else:
            params_ref = None
            for sysname, path in self.modpaths.items():
                with h5py.File(path, "r") as h5:
                    params = h5["parameters"][()]
                    if params_ref is None:
                        params_ref        = params
                        self.parameters   = params
                    elif len(params) != len(params_ref):
                        raise ValueError(f"Row mismatch in {sysname}.")
                    self.h5dict[sysname] = h5[sysname][()]
                    self._systems.append(sysname)

        required = ("logt", "logg", "feh", "afe")
        have     = self.parameters.dtype.names
        missing  = [f for f in required if f not in have]
        if missing:
            raise ValueError(f"/parameters missing fields: {missing}; found: {have}")

        # ---- vmic fixed-detection ----
        # If vmic is present but constant across all rows, treat it as fixed
        # and exclude it from label_i so the network doesn't receive a useless feature.
        self._vmic_is_fixed = False
        self._vmic_fixed_value: Optional[float] = None
        if "vmic" in self.parameters.dtype.names:
            vmic_vals = self.parameters["vmic"].astype(np.float64)
            if np.allclose(vmic_vals, vmic_vals[0], rtol=0, atol=1e-6):
                self._vmic_is_fixed       = True
                self._vmic_fixed_value    = float(vmic_vals[0])
                if self.verbose:
                    print(f"[ReadPhot] vmic is constant ({self._vmic_fixed_value:.4f}); "
                          f"excluded from label_i automatically.")

        # ---- filter mapping ----
        fields_by_system = {s: list(self.h5dict[s].dtype.names) for s in self._systems}
        method       = kwargs.get("filter_wavelength_method", "pivot")
        system_alias = kwargs.get("system_alias", None)
        band_alias   = kwargs.get("band_alias",   None)

        sed_objs: List[SEDpyFilter] = []
        for ff in sedpy_filters:
            if ff.lower().startswith("spherex"):
                sed_objs.append(SEDpyFilter(kname="spherex",
                                            trans_colname=ff.split("_", 1)[1]))
            else:
                sed_objs.extend(observate.load_filters([ff]))
        sed_by_name = {f.name: f for f in sed_objs}

        self.filter_wavelengths: Dict[str, Dict[str, float]] = {}
        self._out_labels: List[str] = []
        self._filter_map: List[Tuple[str, str, str]] = []

        for sname in sedpy_filters:
            f       = sed_by_name[sname]
            wA, T   = _get_filter_arrays(f)
            lamA    = (_pivot_wavelength(wA, T) if method == "pivot"
                       else _logmean_wavelength(wA, T))
            system, band = _resolve_system_band_from_sedpy_name(
                sname, set(self._systems), fields_by_system,
                user_system_alias=system_alias, user_band_alias=band_alias,
            )
            self.filter_wavelengths.setdefault(system, {})[band] = lamA
            self._out_labels.append(f"{system}_{band}")
            self._filter_map.append((system, band, sname))

        # ---- dataset controls ----
        self.datatype    = kwargs.get("type", "train")
        self.returntorch = kwargs.get("returntorch", True)
        self.trainper    = kwargs.get("trainpercentage", 0.9)
        self.norm        = kwargs.get("norm", True)

        default_label_i = ["logt", "logg", "feh", "afe", "av", "rv"]
        # Remove 'vmic' from default if it is fixed (also handles user-supplied lists)
        raw_label_i = kwargs.get("label_i", default_label_i)
        self.label_i: List[str] = [l for l in raw_label_i
                                    if not (l == "vmic" and self._vmic_is_fixed)]
        self.label_o: List[str] = kwargs.get("label_o", self._out_labels)

        self.parrange = kwargs.get("parrange", None)
        self.parameters = rfn.append_fields(
            self.parameters, "model_index",
            np.arange(len(self.parameters)), usemask=False
        )
        if self.parrange is not None:
            for k, (lo, hi) in self.parrange.items():
                if k in self.parameters.dtype.names:
                    self.parameters = self.parameters[
                        (self.parameters[k] >= lo) & (self.parameters[k] <= hi)
                    ]

        # ---- splits ----
        if self.split is not None:
            for key in ("train", "valid", "test"):
                if key not in self.split:
                    raise ValueError(f"split dict missing key '{key}'.")
            mask       = np.isin(self.parameters["model_index"],
                                 self.split[self.datatype])
            base_block = self.parameters[mask]
            self.parameters_train = self.parameters[
                np.isin(self.parameters["model_index"], self.split["train"])]
            self.parameters_valid = self.parameters[
                np.isin(self.parameters["model_index"], self.split["valid"])]
            self.parameters_test  = self.parameters[
                np.isin(self.parameters["model_index"], self.split["test"])]
        else:
            self.rng.shuffle(self.parameters)
            cut        = int(np.rint((1.0 - self.trainper) * len(self.parameters)))
            test_block = self.parameters[:cut]
            rest       = self.parameters[cut:]
            mid        = int(np.rint(0.7 * len(rest)))
            self.parameters_train = rest[:mid]
            self.parameters_valid = rest[mid:]
            self.parameters_test  = test_block
            base_block = {"train": self.parameters_train,
                          "valid": self.parameters_valid,
                          "test":  self.parameters_test}[self.datatype]

        self.split_indices = {
            "train": np.asarray(self.parameters_train["model_index"]),
            "valid": np.asarray(self.parameters_valid["model_index"]),
            "test":  np.asarray(self.parameters_test["model_index"]),
        }

        # ---- extinction ----
        self.extinction_mode = kwargs.get("extinction_mode", None) or (
            "sample" if self.datatype == "train" else "grid"
        )
        self.avgrid = np.array(
            kwargs.get("avgrid",
                       [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                       + list(range(1, 10))
                       + list(range(10, 50, 5))
                       + list(range(50, 101, 10))),
            dtype=np.float32,
        )
        self.rvgrid = np.array(
            kwargs.get("rvgrid", [2.3, 2.5, 3.1, 3.5, 4.0, 5.0, 5.6]),
            dtype=np.float32,
        )
        if self.parrange is not None:
            if "av" in self.parrange:
                lo, hi = self.parrange["av"]
                self.avgrid = self.avgrid[(self.avgrid >= lo) & (self.avgrid <= hi)]
            if "rv" in self.parrange:
                lo, hi = self.parrange["rv"]
                self.rvgrid = self.rvgrid[(self.rvgrid >= lo) & (self.rvgrid <= hi)]
        if len(self.avgrid) == 0:
            raise ValueError("No valid values in avgrid after parrange filtering.")
        if len(self.rvgrid) == 0:
            raise ValueError("No valid values in rvgrid after parrange filtering.")

        self.fixed_av = float(kwargs.get("fixed_av", 0.0))
        self.fixed_rv = float(kwargs.get("fixed_rv", 3.1))
        self.extinction_law = _normalise_extinction_law(
            kwargs.get("extinction_law", kwargs.get("dust_law", "g23"))
        )
        self.extinction_av_break = float(kwargs.get("extinction_av_break", 2.0))
        self.boogert_av_to_ak = float(kwargs.get("boogert_av_to_ak", 1.0 / 7.045))
        self.hybrid_grid_collapse_rv = bool(kwargs.get("hybrid_grid_collapse_rv", True))

        # model_index was appended BEFORE any parrange/split filtering, so it is
        # the original row number in the HDF5 BC arrays.  Use it directly for
        # HDF5 indexing.  Do NOT remap it to the positional row in the filtered
        # self.parameters array; doing so silently pairs each filtered input row
        # with the wrong BC output after parrange cuts.
        base_idx = base_block["model_index"].astype(np.intp)
        if self.extinction_mode == "grid":
            grid_pairs = _build_extinction_grid_pairs(
                self.avgrid, self.rvgrid, self.extinction_law,
                self.extinction_av_break, self.fixed_rv,
                self.hybrid_grid_collapse_rv,
            )
            self._grid_av      = np.array([p[0] for p in grid_pairs], dtype=np.float32)
            self._grid_rv      = np.array([p[1] for p in grid_pairs], dtype=np.float32)
            grid_mult          = len(grid_pairs)
            self._selind       = np.repeat(base_idx, grid_mult).astype(np.intp)
            self._param_rows   = np.repeat(base_block, grid_mult)
            self._per_row_grid = grid_mult
        else:
            self._grid_av      = None
            self._grid_rv      = None
            self._selind       = base_idx.astype(np.intp)
            self._param_rows   = base_block
            self._per_row_grid = 1

        # ---- normalisation ----
        if self.normfactor_override is not None:
            self.normfactor = dict(self.normfactor_override)
        else:
            self.normfactor: Dict[str, Tuple[float, float]] = {}
            for ll in self.label_i:
                if ll in base_block.dtype.names:
                    x = base_block[ll].astype(np.float64)
                elif ll == "av":
                    x = self.avgrid.astype(np.float64)
                elif ll == "rv":
                    x = self.rvgrid.astype(np.float64)
                else:
                    self.normfactor[ll] = (0.0, 1.0)
                    continue
                mu  = float(np.mean(x))
                sdv = float(np.std(x))
                self.normfactor[ll] = (mu, sdv if sdv > 0 else 1.0)
            for lab, (system, band, _) in zip(self.label_o, self._filter_map):
                bc  = self.h5dict[system][band].astype(np.float64)
                mu  = float(np.mean(bc))
                sdv = float(np.std(bc))
                self.normfactor[lab] = (mu, sdv if sdv > 0 else 1.0)

        # ---- k(λ) cache (per-filter scalar; key rounded to avoid float-equality misses) ----
        self._k_cache: Dict[Tuple[str, float, str, str], float] = {}

        self.datalen = len(self._selind)
        if self.verbose:
            print(f"[ReadPhot] type={self.datatype}, ext={self.extinction_mode}, "
                  f"N={self.datalen}, outputs={self.label_o}")

    # ---- helpers ----
    @staticmethod
    def _infer_systems_from_filters(filters: Sequence[str]) -> set:
        systems = set()
        for fname in filters:
            name_lc, matched = fname.lower(), False
            for pref, (sysname, _) in _DEFAULT_PREFIX_MAP.items():
                if re.match(pref, name_lc):
                    systems.add(sysname); matched = True; break
            if not matched:
                parts = fname.split("_", 1)
                systems.add(parts[0].lower() if len(parts) == 2 else fname.lower())
        return systems

    def normf(self, x, label: str):
        mu, sd = self.normfactor[label]
        return (x - mu) / sd

    def unnormf(self, x, label: str):
        mu, sd = self.normfactor[label]
        return x * sd + mu

    def _k_for(self, av: float, rv: float, system: str, band: str) -> float:
        """A(lambda)/A(V) for one filter and the active extinction law."""
        law = _active_extinction_law(
            self.extinction_law, av=av, av_break=self.extinction_av_break
        )
        rv_key = round(float(rv), 4) if law == "g23" else 0.0
        key    = (law, rv_key, system, band)
        if key in self._k_cache:
            return self._k_cache[key]

        lamA = self.filter_wavelengths[system][band]
        if law == "g23":
            lo, hi = 2.3, 5.6
            rvf    = float(np.clip(rv_key, np.nextafter(lo, 10.0), np.nextafter(hi, 0.0)))
            x_inv  = (1.0 / (lamA * 1e-4)) * u.micron ** -1
            k      = float(G23(Rv=rvf)(x_inv))
        elif law == "boogert":
            k      = float(_boogert_k_av(lamA * 1e-4, self.boogert_av_to_ak))
        else:  # pragma: no cover; guarded by _normalise_extinction_law
            raise RuntimeError(f"Unsupported active extinction law: {law}")

        self._k_cache[key] = k
        return k

    # ---- Dataset API ----
    def __len__(self) -> int:
        return self.datalen

    def __getitem__(self, idx: int):
        selind = self._selind[idx]
        row    = self._param_rows[idx]

        if self.extinction_mode == "grid":
            gpos = idx % self._per_row_grid
            av   = float(self._grid_av[gpos])
            rv   = float(self._grid_rv[gpos])
        elif self.extinction_mode == "fixed":
            av, rv = self.fixed_av, self.fixed_rv
        elif self.extinction_mode == "sample":
            av = float(self.rng.choice(self.avgrid))
            rv = float(self.rng.choice(self.rvgrid))
        else:
            av, rv = 0.0, 3.1

        bcout: List[float] = []
        for lab, (system, band, _) in zip(self.label_o, self._filter_map):
            bc = float(self.h5dict[system][band][selind])
            if self.extinction_mode != "none":
                bc = bc - av * self._k_for(av, rv, system, band)
            bcout.append(self.normf(bc, lab) if self.norm else bc)

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


# -----------------------------------------------------------------------
# ReadSpec
# -----------------------------------------------------------------------
class ReadSpec(Dataset):
    """Dataset of synthetic spectra from a Korg spectral grid.

    Parameters
    ----------
    modpath : str | list[str]
        Directory containing *.h5 spectral files, a single .h5 path, or an
        explicit list of paths.
    wave_range : (float, float) | None
        (lo_Å, hi_Å) wavelength window. None = full native grid.
    dlambda : float | None
        Constant Δλ resampling step (Å). Mutually exclusive with *R*.
    R : float | None
        Constant resolving power λ/Δλ for geometric resampling. Mutually
        exclusive with *dlambda*.
    pixels_per_resel : float, default 3.0
        Pixels per resolution element when using *R*.
    rebin_mode : {'interp', 'bin'}, default 'interp'
        Resampling strategy. 'bin' requires *dlambda*.
    use_norm_from_h5 : bool, default True
        Use pre-computed norm/global/raw statistics from HDF5 when no
        resampling is applied (faster; falls back to on-the-fly compute).
    type : {'train', 'valid', 'test'}, default 'train'
    extinction_mode : {'sample', 'grid', 'fixed', 'none'}
        Same semantics as ReadPhot.
    avgrid, rvgrid, fixed_av, fixed_rv
        Same semantics as ReadPhot.
    norm : bool, default False
        Apply z-score normalisation to spectrum outputs. Usually False when
        training in log-flux space.
    continuum_mode : {'none', 'divide'}, default 'none'
        'divide' divides each spectrum by its continuum (requires the HDF5
        to contain a 'continuua' dataset).
    label_i : list[str]
        Input feature names. Default auto-detected from grid
        (['logt','logg','feh','afe'] or + 'vmic' if free, + 'av','rv' if
        extinction phase).
    split, split_seed, trainpercentage, parrange, normfactor, returntorch, verbose
        Same semantics as ReadPhot.

    Notes
    -----
    If all rows in the grid share the same *vmic* value, 'vmic' is silently
    excluded from *label_i* regardless of the user-supplied list, and
    ``self.vmic_is_fixed`` / ``self.vmic_fixed_value`` are set accordingly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs   = kwargs
        self.verbose  = kwargs.get("verbose", False)
        self.progressbar = kwargs.get("progressbar", True)

        self.split_seed = kwargs.get("split_seed", kwargs.get("seed", None))
        self.rng        = np.random.default_rng(self.split_seed)
        self.split      = kwargs.get("split", None)
        self.normfactor_override = kwargs.get("normfactor", None)

        # ---- wavelength controls ----
        self.wave_range      = kwargs.get("wave_range", None)
        self.dlambda         = kwargs.get("dlambda", None)
        self.R               = kwargs.get("R", None)
        self.rebin_mode      = kwargs.get("rebin_mode", "interp")
        self.use_norm_from_h5 = bool(kwargs.get("use_norm_from_h5", True))
        self.pixels_per_resel = kwargs.get("pixels_per_resel", 3.0)

        if self.dlambda is not None and self.R is not None:
            raise ValueError("Provide only one of dlambda or R (not both).")
        if self.rebin_mode not in ("interp", "bin"):
            raise ValueError("rebin_mode must be 'interp' or 'bin'.")
        if self.rebin_mode == "bin" and self.dlambda is None:
            raise ValueError("rebin_mode='bin' requires dlambda.")
        if self.R is not None and (self.pixels_per_resel is None
                                    or self.pixels_per_resel <= 0):
            raise ValueError("pixels_per_resel must be positive when R is set.")

        # ---- discover files ----
        modpath = kwargs.get("modpath", None)
        if modpath is None:
            raise ValueError("Provide `modpath`.")
        if isinstance(modpath, (list, tuple)):
            file_list = [str(p) for p in modpath]
        elif os.path.isdir(modpath):
            file_list = sorted(glob.glob(os.path.join(modpath, "*.h5")))
        elif os.path.isfile(modpath):
            file_list = [modpath]
        else:
            raise FileNotFoundError(f"modpath not found: {modpath}")
        if not file_list:
            raise FileNotFoundError("No .h5 files discovered.")

        # ---- build target wavelength grid ----
        with h5py.File(file_list[0], "r") as h5_0:
            w_native_full = np.array(h5_0["wavelengths"][()], dtype=np.float64)

        self.wavelengths_A, _col_idx = self._build_target_grid(w_native_full)
        resampling   = (_col_idx is None)
        col_idx_keep = None if resampling else _col_idx

        # ---- load and concatenate ----
        params_list, spectra_list, cont_list = [], [], []
        iter_files = (
            tqdm(file_list, desc="Loading spectral files", unit="file",
                 leave=False, disable=not self.verbose)
            if self.progressbar else file_list
        )
        for fp in iter_files:
            with h5py.File(fp, "r") as h5:
                for dset in ("parameters", "wavelengths", "spectra"):
                    if dset not in h5:
                        raise KeyError(f"{fp} missing required dataset '{dset}'.")
                wv = np.array(h5["wavelengths"][()], dtype=np.float64)
                if wv.size != w_native_full.size or not np.allclose(
                        wv, w_native_full, rtol=0, atol=1e-8):
                    raise ValueError(f"Wavelength grid mismatch in {fp}.")
                if not resampling:
                    sp = h5["spectra"][:, col_idx_keep].astype(np.float32)
                    ct = (h5["continuua"][:, col_idx_keep].astype(np.float32)
                          if "continuua" in h5 else None)
                else:
                    sp, ct = self._resample_file(h5, wv)
                params_list.append(h5["parameters"][()])
                spectra_list.append(sp)
                if ct is not None:
                    cont_list.append(ct)

        self.spectra       = np.vstack(spectra_list)
        self.has_continuum = len(cont_list) == len(spectra_list)
        self.continuua     = np.vstack(cont_list) if self.has_continuum else None
        self.parameters    = rfn.stack_arrays(params_list, usemask=False,
                                               asrecarray=False)

        required = ("logt", "logg", "feh", "afe")
        have     = self.parameters.dtype.names
        miss     = [f for f in required if f not in have]
        if miss:
            raise ValueError(f"/parameters missing fields: {miss}; found: {have}")

        # ---- vmic fixed-detection ----
        self.vmic_is_fixed    = False
        self.vmic_fixed_value: Optional[float] = None
        if "vmic" in self.parameters.dtype.names:
            vmic_vals = self.parameters["vmic"].astype(np.float64)
            if np.allclose(vmic_vals, vmic_vals[0], rtol=0, atol=1e-6):
                self.vmic_is_fixed    = True
                self.vmic_fixed_value = float(vmic_vals[0])
                if self.verbose:
                    print(f"[ReadSpec] vmic is constant ({self.vmic_fixed_value:.4f}); "
                          f"excluded from label_i automatically.")

        # ---- dataset controls ----
        self.datatype      = kwargs.get("type", "train")
        self.returntorch   = kwargs.get("returntorch", True)
        self.trainper      = kwargs.get("trainpercentage", 0.9)
        self.norm          = kwargs.get("norm", False)
        self.continuum_mode = kwargs.get("continuum_mode", "none")
        if self.continuum_mode not in ("none", "divide"):
            raise ValueError("continuum_mode must be 'none' or 'divide'.")

        # ---- labels ----
        # Default label_i: physical params (dropping vmic if fixed), then av/rv
        # if the caller includes them.
        default_phys = (["logt", "logg", "feh", "afe"]
                        + ([] if self.vmic_is_fixed else ["vmic"])
                        + ["av", "rv"])
        raw_label_i  = kwargs.get("label_i", default_phys)
        # Silently drop 'vmic' if it's fixed, even if user included it
        self.label_i: List[str] = [l for l in raw_label_i
                                    if not (l == "vmic" and self.vmic_is_fixed)]
        self.label_o: List[str] = [f"lam_{int(round(lam))}"
                                   for lam in self.wavelengths_A]
        self._lam_count = len(self.wavelengths_A)

        # ---- parameter range filtering ----
        self.parrange = kwargs.get("parrange", None)
        self.parameters = rfn.append_fields(
            self.parameters, "model_index",
            np.arange(len(self.parameters)), usemask=False
        )
        if self.parrange is not None:
            mask = np.ones(len(self.parameters), dtype=bool)
            for k, (lo, hi) in self.parrange.items():
                if k in self.parameters.dtype.names:
                    mask &= (self.parameters[k] >= lo) & (self.parameters[k] <= hi)
            if not np.any(mask):
                raise ValueError("parrange filtering eliminated all rows.")
            self.parameters = self.parameters[mask]
            self.spectra    = self.spectra[mask, :]
            if self.has_continuum:
                self.continuua = self.continuua[mask, :]

        # ---- splits ----
        if self.split is not None:
            for key in ("train", "valid", "test"):
                if key not in self.split:
                    raise ValueError(f"split dict missing key '{key}'.")
            mask       = np.isin(self.parameters["model_index"],
                                 self.split[self.datatype])
            base_block = self.parameters[mask]
            self.parameters_train = self.parameters[
                np.isin(self.parameters["model_index"], self.split["train"])]
            self.parameters_valid = self.parameters[
                np.isin(self.parameters["model_index"], self.split["valid"])]
            self.parameters_test  = self.parameters[
                np.isin(self.parameters["model_index"], self.split["test"])]
        else:
            order = np.arange(len(self.parameters))
            self.rng.shuffle(order)
            self.parameters = self.parameters[order]
            self.spectra    = self.spectra[order, :]
            if self.has_continuum:
                self.continuua = self.continuua[order, :]
            cut        = int(np.rint((1.0 - self.trainper) * len(self.parameters)))
            test_block = self.parameters[:cut]
            rest       = self.parameters[cut:]
            mid        = int(np.rint(0.7 * len(rest)))
            self.parameters_train = rest[:mid]
            self.parameters_valid = rest[mid:]
            self.parameters_test  = test_block
            base_block = {"train": self.parameters_train,
                          "valid": self.parameters_valid,
                          "test":  self.parameters_test}[self.datatype]

        self.split_indices = {
            "train": np.asarray(self.parameters_train["model_index"]),
            "valid": np.asarray(self.parameters_valid["model_index"]),
            "test":  np.asarray(self.parameters_test["model_index"]),
        }

        # ---- extinction ----
        self.extinction_mode = kwargs.get("extinction_mode", None) or (
            "sample" if self.datatype == "train" else "grid"
        )
        self.avgrid = np.array(
            kwargs.get("avgrid",
                       [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                       + list(range(1, 10))
                       + list(range(10, 50, 5))
                       + list(range(50, 101, 10))),
            dtype=np.float32,
        )
        self.rvgrid = np.array(
            kwargs.get("rvgrid", [2.3, 2.5, 3.1, 3.5, 4.0, 5.0, 5.6]),
            dtype=np.float32,
        )
        if self.parrange is not None:
            if "av" in self.parrange:
                lo, hi = self.parrange["av"]
                self.avgrid = self.avgrid[(self.avgrid >= lo) & (self.avgrid <= hi)]
            if "rv" in self.parrange:
                lo, hi = self.parrange["rv"]
                self.rvgrid = self.rvgrid[(self.rvgrid >= lo) & (self.rvgrid <= hi)]
        if len(self.avgrid) == 0:
            raise ValueError("No valid values in avgrid after parrange filtering.")
        if len(self.rvgrid) == 0:
            raise ValueError("No valid values in rvgrid after parrange filtering.")

        self.fixed_av = float(kwargs.get("fixed_av", 0.0))
        self.fixed_rv = float(kwargs.get("fixed_rv", 3.1))
        self.extinction_law = _normalise_extinction_law(
            kwargs.get("extinction_law", kwargs.get("dust_law", "g23"))
        )
        self.extinction_av_break = float(kwargs.get("extinction_av_break", 2.0))
        self.boogert_av_to_ak = float(kwargs.get("boogert_av_to_ak", 1.0 / 7.045))
        self.hybrid_grid_collapse_rv = bool(kwargs.get("hybrid_grid_collapse_rv", True))

        # model_index was appended BEFORE any parrange/split filtering, so it is
        # the original row number in the HDF5 BC arrays.  Use it directly for
        # HDF5 indexing.  Do NOT remap it to the positional row in the filtered
        # self.parameters array; doing so silently pairs each filtered input row
        # with the wrong BC output after parrange cuts.
        base_idx = base_block["model_index"].astype(np.intp)
        if self.extinction_mode == "grid":
            grid_pairs = _build_extinction_grid_pairs(
                self.avgrid, self.rvgrid, self.extinction_law,
                self.extinction_av_break, self.fixed_rv,
                self.hybrid_grid_collapse_rv,
            )
            self._grid_av      = np.array([p[0] for p in grid_pairs], dtype=np.float32)
            self._grid_rv      = np.array([p[1] for p in grid_pairs], dtype=np.float32)
            grid_mult          = len(grid_pairs)
            self._selind       = np.repeat(base_idx, grid_mult).astype(np.intp)
            self._param_rows   = np.repeat(base_block, grid_mult)
            self._per_row_grid = grid_mult
        else:
            self._grid_av      = None
            self._grid_rv      = None
            self._selind       = base_idx.astype(np.intp)
            self._param_rows   = base_block
            self._per_row_grid = 1

        # ---- normalisation ----
        if self.normfactor_override is not None:
            self.normfactor = dict(self.normfactor_override)
        else:
            self.normfactor: Dict[str, Tuple[float, float]] = {}
            for ll in self.label_i:
                if ll in self.parameters.dtype.names:
                    x = self.parameters[ll].astype(np.float64)
                elif ll == "av":
                    x = self.avgrid.astype(np.float64)
                elif ll == "rv":
                    x = self.rvgrid.astype(np.float64)
                else:
                    self.normfactor[ll] = (0.0, 1.0)
                    continue
                mu  = float(np.mean(x))
                sdv = float(np.std(x))
                self.normfactor[ll] = (mu, sdv if sdv > 0 else 1.0)

            # Output normalisation: try H5 pre-computed stats first
            could_use_h5 = self.use_norm_from_h5 and not resampling
            mu_vec = sd_vec = None
            if could_use_h5:
                try:
                    with h5py.File(file_list[0], "r") as h5:
                        g      = h5["norm/global/raw"]
                        mu_vec = np.array(g["mean_spectrum"][()],
                                          dtype=np.float64)[col_idx_keep]
                        sd_vec = np.array(g["std_spectrum"][()],
                                          dtype=np.float64)[col_idx_keep]
                except Exception:
                    mu_vec = sd_vec = None

            if mu_vec is None or sd_vec is None or mu_vec.size != self._lam_count:
                base_mask = np.isin(self.parameters["model_index"], base_idx)
                Y = self.spectra[base_mask, :].astype(np.float64)
                if self.continuum_mode == "divide" and self.has_continuum:
                    Y = Y / np.maximum(self.continuua[base_mask, :], 1e-30)
                mu_vec = np.mean(Y, axis=0)
                sd_vec = np.std(Y, axis=0)
                sd_vec[sd_vec <= 0] = 1.0

            for j, lab in enumerate(self.label_o):
                self.normfactor[lab] = (float(mu_vec[j]), float(sd_vec[j]))

        # ---- k(λ) cache for vector extinction (key rounded to avoid float-equality misses) ----
        self._k_lambda_cache: Dict[Tuple[str, float], np.ndarray] = {}
        self._x_inv_micron   = (1.0 / (self.wavelengths_A * 1e-4)) * u.micron ** -1
        self._wavelengths_micron = self.wavelengths_A * 1e-4

        self.datalen = len(self._selind)
        if self.verbose:
            print(f"[ReadSpec] type={self.datatype}, ext={self.extinction_mode}, "
                  f"N={self.datalen}, L={self._lam_count}, "
                  f"vmic_fixed={self.vmic_is_fixed}")

    # ---- wavelength grid helpers ----
    def _build_target_grid(
        self, native_w: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (lambda_target, col_idx_native_or_None)."""
        w = native_w
        if self.wave_range is not None:
            lo, hi = map(float, self.wave_range)
            lo = max(lo, float(w[0]))
            hi = min(hi, float(w[-1]))
            if hi <= lo:
                raise ValueError(f"wave_range {self.wave_range} outside native grid.")
            mask = (w >= lo) & (w <= hi)
            w        = w[mask]
            col_idx  = np.nonzero(mask)[0]
        else:
            col_idx  = np.arange(len(w), dtype=int)

        if self.dlambda is None and self.R is None:
            return w, col_idx          # no resampling

        loA, hiA = float(w[0]), float(w[-1])
        if self.dlambda is not None:
            step  = float(self.dlambda)
            n     = int(np.floor((hiA - loA) / step)) + 1
            lam_t = loA + step * np.arange(n, dtype=np.float64)
            if lam_t[-1] < hiA:
                lam_t = np.append(lam_t, hiA)
        else:
            f     = 1.0 + 1.0 / (float(self.R) * float(self.pixels_per_resel))
            lam   = [loA]
            while lam[-1] < hiA:
                nxt = lam[-1] * f
                lam.append(nxt if nxt < hiA else hiA)
            lam_t = np.array(lam, dtype=np.float64)

        return lam_t, None

    def _resample_file(
        self, h5: h5py.File, wv: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Interpolate (or bin) one file's spectra onto self.wavelengths_A."""
        if self.wave_range is not None:
            lo, hi   = map(float, self.wave_range)
            mask_nat = (wv >= max(lo, wv[0])) & (wv <= min(hi, wv[-1]))
            cols     = np.nonzero(mask_nat)[0]
        else:
            cols = slice(None)

        w_slice    = wv[cols]
        sp_native  = h5["spectra"][:, cols].astype(np.float64)
        ct_native  = (h5["continuua"][:, cols].astype(np.float64)
                      if "continuua" in h5 else None)
        lam_t      = self.wavelengths_A
        N          = sp_native.shape[0]

        sp = np.empty((N, lam_t.size), dtype=np.float32)
        for i in range(N):
            sp[i, :] = np.interp(lam_t, w_slice, sp_native[i, :])

        ct = None
        if ct_native is not None:
            ct = np.empty((N, lam_t.size), dtype=np.float32)
            for i in range(N):
                ct[i, :] = np.interp(lam_t, w_slice, ct_native[i, :])

        return sp, ct

    # ---- extinction vector cache ----
    def _k_lambda(self, av: float, rv: float) -> np.ndarray:
        """A(lambda)/A(V) vector for the active extinction law."""
        law = _active_extinction_law(
            self.extinction_law, av=av, av_break=self.extinction_av_break
        )
        rv_key = round(float(rv), 4) if law == "g23" else 0.0
        key = (law, rv_key)
        if key in self._k_lambda_cache:
            return self._k_lambda_cache[key]

        if law == "g23":
            lo, hi = 2.3, 5.6
            rvf    = float(np.clip(rv_key, np.nextafter(lo, 10.0),
                                    np.nextafter(hi, 0.0)))
            kvec   = np.array(G23(Rv=rvf)(self._x_inv_micron), dtype=np.float64)
        elif law == "boogert":
            kvec   = np.array(
                _boogert_k_av(self._wavelengths_micron, self.boogert_av_to_ak),
                dtype=np.float64,
            )
        else:  # pragma: no cover; guarded by _normalise_extinction_law
            raise RuntimeError(f"Unsupported active extinction law: {law}")

        self._k_lambda_cache[key] = kvec
        return kvec

    # ---- normalisation helpers ----
    def normf(self, x, label: str):
        mu, sd = self.normfactor[label]
        return (x - mu) / sd

    def unnormf(self, x, label: str):
        mu, sd = self.normfactor[label]
        return x * sd + mu

    # ---- Dataset API ----
    def __len__(self) -> int:
        return self.datalen

    def __getitem__(self, idx: int):
        selind = self._selind[idx]
        row    = self._param_rows[idx]

        if self.extinction_mode == "grid":
            gpos = idx % self._per_row_grid
            av   = float(self._grid_av[gpos])
            rv   = float(self._grid_rv[gpos])
        elif self.extinction_mode == "fixed":
            av, rv = self.fixed_av, self.fixed_rv
        elif self.extinction_mode == "sample":
            av = float(self.rng.choice(self.avgrid))
            rv = float(self.rng.choice(self.rvgrid))
        else:
            av, rv = 0.0, 3.1

        y = self.spectra[selind, :].astype(np.float64)
        if self.continuum_mode == "divide" and self.has_continuum:
            y = y / np.maximum(self.continuua[selind, :].astype(np.float64), 1e-30)
        if self.extinction_mode != "none":
            y = y * np.power(10.0, -0.4 * av * self._k_lambda(av, rv))

        y_out = self.normf(y, self.label_o) if self.norm else y

        x_list: List[float] = []
        for ll in self.label_i:
            if ll in row.dtype.names:
                val = float(row[ll])
            elif ll == "av":
                val = av
            elif ll == "rv":
                val = rv
            else:
                raise KeyError(f"Input label '{ll}' not found in parameters or av/rv.")
            x_list.append(self.normf(val, ll) if self.norm else val)

        flat = np.concatenate(
            [np.asarray(x_list, dtype=np.float64), np.asarray(y_out, dtype=np.float64)]
        ).astype(np.float32)
        return torch.tensor(flat) if self.returntorch else flat


# -----------------------------------------------------------------------
# XYFromFlat
# -----------------------------------------------------------------------
class XYFromFlat(torch.utils.data.Dataset):
    """Split a flat-vector ReadPhot / ReadSpec dataset into (x, y) pairs.

    Parameters
    ----------
    base_ds : ReadPhot | ReadSpec
        The underlying dataset whose items are flat [x || y] vectors.

    Returns
    -------
    (x, y) : (Tensor, Tensor)
        x has length ``len(base_ds.label_i)``,
        y has length ``len(base_ds.label_o)``.
    """

    def __init__(self, base_ds) -> None:
        self.ds    = base_ds
        self.n_in  = len(base_ds.label_i)
        self.n_out = len(base_ds.label_o)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        flat = self.ds[idx]
        return flat[: self.n_in], flat[self.n_in :]