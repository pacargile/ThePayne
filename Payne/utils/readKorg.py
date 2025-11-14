"""readKorg

Production-ready dataloader for bolometric-correction (BC) tables produced
from Korg/GenPhot, with on-the-fly G23 extinction and SEDpy-driven
filter wavelength handling.

Key features
------------
- Supports single HDF5 (with multiple photometric systems as datasets) or a
  dict of per-system HDF5 files.
- Robust mapping from SEDpy filter names (e.g., 'gaia_g', 'ps_g', 'roman_wfi_f062')
  to HDF5 (system, band) fields via rules + optional aliases.
- Representative wavelength per filter via SEDpy (pivot or log-mean).
- Extinction on-the-fly using Gordon+23 (G23) with safe clamping of R_V.
- Deterministic train/valid/test splitting or externally supplied splits.
- Optional z-score normalization for inputs and outputs (computed from
  intrinsic BCs).

Public API
----------
- ReadPhot: torch.utils.data.Dataset yielding a flat vector [x_in || y_out].
- XYFromFlat: thin wrapper that returns (x, y) tensors from a ReadPhot.

Notes
-----
- Normalization statistics can be overridden via `normfactor` to ensure
  compatibility with previously trained models.
- Extinction modes: 'sample' (train default), 'grid' (valid/test default),
  'fixed', 'none'. Grid mode expands each base row by all (A_V, R_V) pairs.
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


__all__ = ["ReadPhot", "ReadSpec", "XYFromFlat"]

# ------------------------------
# Optional dependency: sedpy
# ------------------------------
try:
    from sedpy import observate
    from sedpy.observate import Filter as SEDpyFilter
except Exception as e:  # pragma: no cover
    raise ImportError("`sedpy` is required when using `filters`.") from e

# ------------------------------
# SEDpy wavelength helpers (inlined for speed & robustness)
# ------------------------------
def _pivot_wavelength(wave_A: np.ndarray, trans: np.ndarray) -> float:
    """SEDpy-like pivot wavelength in Angstrom.

    λ_p = sqrt(∫ S(λ) λ dλ / ∫ S(λ) / λ dλ)
    """
    w = np.asarray(wave_A, dtype=float)
    S = np.asarray(trans, dtype=float)
    num = np.trapz(S * w, w)
    den = np.trapz(S / w, w)
    return float(np.sqrt(num / den))


def _logmean_wavelength(wave_A: np.ndarray, trans: np.ndarray) -> float:
    """Log-mean wavelength (Angstrom).

    exp(Σ ln λ * S * dlnλ / Σ S * dlnλ)
    """
    w = np.asarray(wave_A, dtype=float)
    S = np.asarray(trans, dtype=float)
    lnw = np.log(w)
    dlnw = np.gradient(lnw)
    return float(np.exp(np.sum(lnw * S * dlnw) / np.sum(S * dlnw)))


# ------------------------------
# Filesystem utilities
# ------------------------------
def _as_dict_of_paths(modpath: str | Mapping[str, str]) -> Dict[str, str]:
    """Normalize `modpath` to a {system: path} dictionary.

    - If a directory is provided, it collects all '*.h5' under it and uses the
      stem as the system name.
    - If a single file is provided, uses key '__single__'.
    - If a dict is provided, validates that files exist.
    """
    if isinstance(modpath, dict):
        out = {k: v for k, v in modpath.items()}
        for p in out.values():
            if not os.path.isfile(p):
                raise FileNotFoundError(f"HDF5 file not found: {p}")
        return out

    if os.path.isdir(modpath):
        out: Dict[str, str] = {}
        for p in sorted(glob.glob(os.path.join(modpath, "*.h5"))):
            sysname = os.path.splitext(os.path.basename(p))[0]
            out[sysname] = p
        if not out:
            raise FileNotFoundError(f"No .h5 under: {modpath}")
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
    """Return (wave_A, trans) from a SEDpy Filter object, robust to API variants."""
    wave = _first_attr(f, ("wave", "wavelength", "lam", "_wavelength"))
    trans = _first_attr(f, ("trans", "throughput", "_transmission"))
    if wave is None or trans is None:
        raise TypeError(
            "Could not find wavelength/throughput on sedpy Filter. "
            "Tried wave|wavelength|lam|_wavelength and trans|throughput|_transmission."
        )
    return np.asarray(wave, dtype=float), np.asarray(trans, dtype=float)


# ------------------------------
# SEDpy-name → (system, band) resolver
# ------------------------------
_DEFAULT_PREFIX_MAP: Dict[str, Tuple[str, callable]] = {
    r"^ps_": ("panstarrs", lambda name: name.split("_", 1)[1]),
    r"^gaia_": ("gaia", lambda name: name.split("_", 1)[1]),
    r"^twomass_": ("twomass", lambda name: name.split("_", 1)[1]),
    r"^wise_": ("wise", lambda name: name.split("_", 1)[1]),
    r"^sdss_": ("sdss", lambda name: name.split("_", 1)[1].rstrip("0")),
    r"^decam_": ("decam", lambda name: name.split("_", 1)[1]),
    r"^lsst_": ("lsst", lambda name: name.split("_", 1)[1]),
    r"^roman_": ("roman_wfi", lambda name: name.split("_", 2)[2]),
    r"^swift_": ("uvot", lambda name: name.split("_", 1)[1]),
    r"^spx_": ("spherex", lambda name: name.split("_", 1)[1]),
}


def _resolve_system_band_from_sedpy_name(
    sedpy_name: str,
    h5_system_names: Iterable[str],
    h5_fields_by_system: Mapping[str, Sequence[str]],
    user_system_alias: Optional[Mapping[str, str]] = None,
    user_band_alias: Optional[Mapping[Tuple[str, str], str]] = None,
    ) -> Tuple[str, str]:
    """Map a SEDpy filter name to (system, band) used in the HDF5.

    - Uses _DEFAULT_PREFIX_MAP (incl. Roman WFI).
    - Applies optional user aliases.
    - Adapts band 'case' automatically to match HDF5 field names.
    """
    name_lc = sedpy_name.lower()

    # 1) Rule-based inference of system + band guess
    system: Optional[str] = None
    band_guess: Optional[str] = None
    for pref, (sysname, band_fn) in _DEFAULT_PREFIX_MAP.items():
        if re.match(pref, name_lc):
            system = sysname
            band_guess = band_fn(sedpy_name)
            break
    if system is None or band_guess is None:
        parts = sedpy_name.split("_", 1)
        system = parts[0].lower() if len(parts) == 2 else sedpy_name.lower()
        band_guess = parts[1] if len(parts) == 2 else sedpy_name

    # 2) Apply user aliases (optional)
    if user_system_alias and system in user_system_alias:
        system = user_system_alias[system]
    if user_band_alias and (system, band_guess) in user_band_alias:
        band_guess = user_band_alias[(system, band_guess)]

    # 3) Snap system to a real HDF5 system name if needed
    h5_system_names = list(h5_system_names)
    if system not in h5_system_names:
        cand = [s for s in h5_system_names if s.lower() == system.lower() or s.lower().startswith(system)]
        if len(cand) == 1:
            system = cand[0]
        elif len(cand) > 1:
            system = sorted(cand, key=len, reverse=True)[0]
        else:
            raise KeyError(
                f"System '{system}' (from '{sedpy_name}') not found in HDF5 systems {sorted(h5_system_names)}"
            )

    # 4) Snap band to an exact HDF5 field (case-insensitive, tolerant to small style diffs)
    fields = list(h5_fields_by_system.get(system, []))
    if not fields:
        raise KeyError(f"No filter fields found for system '{system}' in HDF5.")

    if band_guess in fields:
        return system, band_guess

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    fields_norm = {_norm(f): f for f in fields}
    band_norm = _norm(band_guess)

    ci_map = {f.lower(): f for f in fields}
    if band_guess.lower() in ci_map:
        return system, ci_map[band_guess.lower()]
    if band_norm in fields_norm:
        return system, fields_norm[band_norm]

    hits = [orig for norm, orig in fields_norm.items() if norm.startswith(band_norm) or band_norm.startswith(norm)]
    if len(hits) == 1:
        return system, hits[0]

    raise KeyError(
        f"Cannot map SEDpy filter '{sedpy_name}' to HDF5 ({system}, '{band_guess}'). "
        f"Available fields in '{system}': {sorted(fields)}"
    )


class ReadPhot(Dataset):
    """Dataset of synthetic photometry with optional extinction.

    Parameters
    ----------
    modpath : str | dict
        HDF5 path (single-file or directory) or {system: path} mapping.
    filters : list[str]
        SEDpy filter names (e.g., ['ps_g','ps_r','gaia_g', ...]).
    filter_wavelength_method : {'pivot','logmean'}, default 'pivot'
        Method for representative wavelength.
    system_alias, band_alias : dict, optional
        Aliases for resolving (system, band) from SEDpy names.
    type : {'train','valid','test'}, default 'train'
    extinction_mode : {'sample','grid','fixed','none'}
        Train default is 'sample'; valid/test default is 'grid'.
    avgrid, rvgrid : Sequence[float]
        Grids for A_V and R_V.
    fixed_av, fixed_rv : float
        Used when extinction_mode == 'fixed'.
    norm : bool, default True
        Apply z-score normalization to inputs/outputs.
    normfactor : Mapping[str, Tuple[float, float]], optional
        Override normalization stats for labels in label_i + label_o.
    split : dict[str, np.ndarray], optional
        External split indices by 'train'/'valid'/'test' over the *model_index* field.
    split_seed : int, optional
        RNG seed for deterministic splits when `split` not provided.
    trainpercentage : float, default 0.9
        Fraction of data used for train+valid when auto-splitting.
    label_i : list[str]
        Input label names; default ['logt','logg','feh','afe','av','rv'].
    label_o : list[str]
        Output label names; default constructed from requested filters.
    returntorch : bool, default True
        If True, returns torch.tensor; else returns np.ndarray.
    verbose : bool, default False
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 (torch-style init)
        super().__init__()
        self.kwargs = kwargs
        self.verbose: bool = kwargs.get("verbose", False)

        # --- reproducible splits + optional external stats ---
        self.split_seed = kwargs.get("split_seed", kwargs.get("seed", None))
        self.rng = np.random.default_rng(self.split_seed)
        self.split: Optional[Dict[str, np.ndarray]] = kwargs.get("split", None)
        self.normfactor_override: Optional[Mapping[str, Tuple[float, float]]] = kwargs.get("normfactor", None)

        # ---------------- Sources ----------------
        modpath = kwargs.get("modpath", None)
        if modpath is None:
            raise ValueError("Provide `modpath` (file, directory, or dict)")
        self.modpaths = _as_dict_of_paths(modpath)

        self.h5dict: Dict[str, np.ndarray] = {}
        self._systems: List[str] = []

        # ------------- Load HDF5s -------------
        if "__single__" in self.modpaths:
            path = self.modpaths["__single__"]
            if self.verbose:
                print(f"... Reading (single-file) {path}")
            with h5py.File(path, "r") as h5:
                # ----- parameters (required) -----
                if "parameters" not in h5 or not isinstance(h5["parameters"], h5py.Dataset):
                    raise KeyError("HDF5 must contain a dataset '/parameters'.")
                self.parameters = h5["parameters"][()]

                expected = ("logt", "logg", "feh", "afe", "vmic")
                have = self.parameters.dtype.names
                missing = [f for f in expected if f not in have]
                if missing:
                    raise ValueError(f"/parameters missing fields: {missing}; found: {have}")

                # ----- optional meta & rowkey -----
                self.meta = dict(h5["meta"].attrs) if "meta" in h5 else {}
                self.rowkey = h5["rowkey"][()] if "rowkey" in h5 and isinstance(h5["rowkey"], h5py.Dataset) else None
                if self.rowkey is not None and len(self.rowkey) != len(self.parameters):
                    raise ValueError(
                        f"/rowkey length {len(self.rowkey)} != /parameters length {len(self.parameters)}"
                    )

                # Determine requested systems from the provided filters
                sedpy_filters = kwargs.get("filters", None)
                if not sedpy_filters:
                    raise ValueError("Pass `filters` (list of filter names).")

                requested_systems = set()
                for fname in sedpy_filters:
                    name_lc = fname.lower()
                    matched = False
                    for pref, (sysname, _band_fn) in _DEFAULT_PREFIX_MAP.items():
                        if re.match(pref, name_lc):
                            requested_systems.add(sysname)
                            matched = True
                            break
                    if not matched:
                        parts = fname.split("_", 1)
                        if len(parts) == 2:
                            requested_systems.add(parts[0].lower())

                NON_SYSTEM_KEYS = {"parameters", "meta", "rowkey"}
                available_systems = {k for k, v in h5.items() if isinstance(v, h5py.Dataset) and k not in NON_SYSTEM_KEYS}
                missing_sys = sorted(requested_systems - available_systems)
                if missing_sys:
                    raise KeyError(
                        f"Requested systems {missing_sys} not found. Available systems in file: {sorted(available_systems)}"
                    )

                for sysname in sorted(requested_systems):
                    ds = h5[sysname]
                    if ds.dtype.names is None:
                        raise TypeError(
                            f"Dataset '/{sysname}' must be a structured array with one field per band."
                        )
                    self.h5dict[sysname] = ds[()]  # load structured array
                    self._systems.append(sysname)
        else:
            params_ref = None
            for sysname, path in self.modpaths.items():
                if self.verbose:
                    print(f"... Reading {sysname} from {path}")
                with h5py.File(path, "r") as h5:
                    params = h5["parameters"][()]
                    phot = h5[sysname][()]
                    if params_ref is None:
                        params_ref = params
                        self.parameters = params
                    else:
                        if len(params) != len(params_ref):
                            raise ValueError(f"Row mismatch in {sysname}: {len(params)} vs {len(params_ref)}")
                    self.h5dict[sysname] = phot
                    self._systems.append(sysname)

        # Sanity on parameters schema
        expected = ("logt", "logg", "feh", "afe", "vmic")
        have = self.parameters.dtype.names
        missing = [f for f in expected if f not in have]
        if missing:
            raise ValueError(f"/parameters missing required fields: {missing}; found: {have}")

        # ----------- Discover available bands per system -----------
        fields_by_system: Dict[str, List[str]] = {sys: list(self.h5dict[sys].dtype.names) for sys in self._systems}

        # ----------- SEDpy filters list → wavelengths + mapping -----------
        sedpy_filters = kwargs.get("filters", None)
        if not sedpy_filters:
            raise ValueError("Pass `filters` as a list of SEDpy filter names (e.g., ['ps_g','gaia_g', ...])")

        method = kwargs.get("filter_wavelength_method", "pivot")
        system_alias = kwargs.get("system_alias", None)
        band_alias = kwargs.get("band_alias", None)

        sed_list = list(sedpy_filters)
        sed_objs: List[SEDpyFilter] = []
        for ff in sed_list:
            if ff.lower().startswith("spherex"):
                # e.g., 'spherex_ch062' → kname='spherex', trans_colname='ch062'
                sed_objs.append(SEDpyFilter(kname="spherex", trans_colname=ff.split("_", 1)[1]))
            else:
                sed_objs.extend(observate.load_filters([ff]))
        sed_by_name = {f.name: f for f in sed_objs}

        self.filter_wavelengths: Dict[str, Dict[str, float]] = {}
        self._out_labels: List[str] = []
        self._filter_map: List[Tuple[str, str, str]] = []  # (system, band, sedpy_name)

        for sname in sed_list:
            f = sed_by_name[sname]
            wA, T = _get_filter_arrays(f)
            if method == "pivot":
                lamA = _pivot_wavelength(wA, T)
            elif method == "logmean":
                lamA = _logmean_wavelength(wA, T)
            else:
                raise ValueError("filter_wavelength_method must be 'pivot' or 'logmean'")

            system, band = _resolve_system_band_from_sedpy_name(
                sname,
                set(self._systems),
                fields_by_system,
                user_system_alias=system_alias,
                user_band_alias=band_alias,
            )
            self.filter_wavelengths.setdefault(system, {})[band] = lamA
            self._out_labels.append(f"{system}_{band}")
            self._filter_map.append((system, band, sname))

        # ------------- Dataset type / splits -------------
        self.datatype: str = kwargs.get("type", "train")
        self.returntorch: bool = kwargs.get("returntorch", True)
        self.trainper: float = kwargs.get("trainpercentage", 0.9)
        self.norm: bool = kwargs.get("norm", True)

        # ------------- Labels -------------
        default_label_i = ["logt", "logg", "feh", "afe", "av", "rv"]
        self.label_i: List[str] = kwargs.get("label_i", default_label_i)
        self.label_o: List[str] = kwargs.get("label_o", self._out_labels)

        if self.normfactor_override is not None:
            missing = [k for k in (self.label_i + self.label_o) if k not in self.normfactor_override]
            if missing:
                raise ValueError(f"normfactor override missing labels: {missing}")

        # ------------- Parameter selection -------------
        self.parrange: Optional[Mapping[str, Tuple[float, float]]] = kwargs.get("parrange", None)
        self.parameters = rfn.append_fields(
            self.parameters, "model_index", np.arange(len(self.parameters)), usemask=False
        )
        if self.parrange is not None:
            for k, (lo, hi) in self.parrange.items():
                if self.verbose:
                    print(f"... Applying parameter range for {k}: [{lo},{hi}]")
                if k in self.parameters.dtype.names:
                    self.parameters = self.parameters[(self.parameters[k] >= lo) & (self.parameters[k] <= hi)]
                # Ignore keys not in parameters (e.g., 'av','rv')

        # --- splits ---
        if self.split is not None:
            for key in ("train", "valid", "test"):
                if key not in self.split:
                    raise ValueError(f"split dict missing key '{key}'")

            want = self.split.get(self.datatype, None)
            if want is None:
                raise ValueError(f"split dict missing key '{self.datatype}'")
            mask = np.isin(self.parameters["model_index"], want)
            base_block = self.parameters[mask]

            self.parameters_train = self.parameters[np.isin(self.parameters["model_index"], self.split.get("train", []))]
            self.parameters_valid = self.parameters[np.isin(self.parameters["model_index"], self.split.get("valid", []))]
            self.parameters_test = self.parameters[np.isin(self.parameters["model_index"], self.split.get("test", []))]
        else:
            self.rng.shuffle(self.parameters)
            cut = int(np.rint((1.0 - self.trainper) * len(self.parameters)))
            test_block = self.parameters[:cut]
            rest = self.parameters[cut:]
            mid = int(np.rint(0.7 * len(rest)))

            train_block = rest[:mid]
            valid_block = rest[mid:]

            self.parameters_train = train_block
            self.parameters_valid = valid_block
            self.parameters_test = test_block

            base_block = {"train": train_block, "valid": valid_block, "test": test_block}[self.datatype]

        self.split_indices = {
            "train": np.asarray(self.parameters_train["model_index"]) if hasattr(self, "parameters_train") else np.array([], dtype=int),
            "valid": np.asarray(self.parameters_valid["model_index"]) if hasattr(self, "parameters_valid") else np.array([], dtype=int),
            "test": np.asarray(self.parameters_test["model_index"]) if hasattr(self, "parameters_test") else np.array([], dtype=int),
        }

        # ------------- Extinction control -------------
        self.extinction_mode: str = kwargs.get("extinction_mode", None) or (
            "sample" if self.datatype == "train" else "grid"
        )

        self.avgrid = np.array(
            kwargs.get(
                "avgrid",
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                + list(range(1, 10, 1))
                + list(range(10, 50, 5))
                + list(range(50, 101, 10)),
            ),
            dtype=np.float32,
        )
        self.rvgrid = np.array(kwargs.get("rvgrid", [2.3, 2.5, 3.1, 3.5, 4.0, 5.0, 5.6]), dtype=np.float32)

        if self.parrange is not None and "av" in self.parrange:
            lo, hi = self.parrange["av"]
            self.avgrid = self.avgrid[(self.avgrid >= lo) & (self.avgrid <= hi)]
        if self.parrange is not None and "rv" in self.parrange:
            lo, hi = self.parrange["rv"]
            self.rvgrid = self.rvgrid[(self.rvgrid >= lo) & (self.rvgrid <= hi)]

        if len(self.avgrid) == 0:
            raise ValueError("No valid values found in avgrid")
        if len(self.rvgrid) == 0:
            raise ValueError("No valid values found in rvgrid")

        self.fixed_av: float = float(kwargs.get("fixed_av", 0.0))
        self.fixed_rv: float = float(kwargs.get("fixed_rv", 3.1))

        base_idx = base_block["model_index"]
        if self.extinction_mode == "grid":
            grid_mult = int(len(self.avgrid) * len(self.rvgrid))
            self._selind = np.repeat(base_idx, grid_mult)
            self._param_rows = np.repeat(base_block, grid_mult)
            self._per_row_grid = grid_mult  # for fast modulo arithmetic
        else:
            self._selind = base_idx
            self._param_rows = base_block
            self._per_row_grid = 1

        # ------------- Normalization (intrinsic BCs) -------------
        if self.normfactor_override is not None:
            self.normfactor = dict(self.normfactor_override)
        else:
            self.normfactor: Dict[str, Tuple[float, float]] = {}
            # Inputs
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
                mu = float(np.mean(x))
                sdv = float(np.std(x))
                sd = sdv if sdv > 0 else 1.0
                self.normfactor[ll] = (mu, sd)

            # Outputs (intrinsic, unreddened)
            for lab, (system, band, _sedname) in zip(self.label_o, self._filter_map):
                bc_arr = self.h5dict[system][band].astype(np.float64)
                mu = float(np.mean(bc_arr))
                sdv = float(np.std(bc_arr))
                sd = sdv if sdv > 0 else 1.0
                self.normfactor[lab] = (mu, sd)

        # ------------- k(λ) cache -------------
        self._k_cache: Dict[Tuple[float, str, str], float] = {}
        self.datalen = len(self._selind)

        if self.verbose:
            print(f"... Data Set Type: {self.datatype}")
            print(f"... Extinction mode: {self.extinction_mode}")
            print(f"... N rows (effective): {self.datalen}")
            print(f"... Systems present: {self._systems}")
            print(f"... Outputs: {self.label_o}")

    # ------------- helpers -------------
    def normf(self, x: float, label: str) -> float:
        mu, sd = self.normfactor[label]
        return (x - mu) / sd

    def unnormf(self, x: float, label: str) -> float:
        mu, sd = self.normfactor[label]
        return x * sd + mu

    def _k_for(self, rv: float, system: str, band: str) -> float:
        key = (float(rv), system, band)
        if key in self._k_cache:
            return self._k_cache[key]

        lamA = self.filter_wavelengths[system][band]  # Å

        # Clamp R_V to (2.3, 5.6) open interval to avoid FP boundary issues
        lo, hi = 2.3, 5.6
        rvf = float(rv)
        if rvf <= lo:
            rvf = float(np.nextafter(lo, 10.0))
        elif rvf >= hi:
            rvf = float(np.nextafter(hi, 0.0))

        x_inv_micron = (1.0 / (lamA * 1e-4)) * u.micron ** -1  # 1/μm
        k = float(G23(Rv=rvf)(x_inv_micron))  # A(λ)/A(V)
        self._k_cache[key] = k
        return k

    def _bc_with_extinction(self, bc_intrinsic: float, system: str, band: str, av: float, rv: float) -> float:
        if self.extinction_mode == "none":
            return bc_intrinsic
        return bc_intrinsic - av * self._k_for(rv, system, band)

    # ------------- Dataset API -------------
    def __len__(self) -> int:
        return self.datalen

    def __getitem__(self, idx: int):
        selind = self._selind[idx]
        row = self._param_rows[idx]

        if self.extinction_mode == "grid":
            # Decode which (A_V, R_V) pair applies to this index
            gpos = idx % self._per_row_grid
            n_rv = len(self.rvgrid)
            av = float(self.avgrid[gpos // n_rv])
            rv = float(self.rvgrid[gpos % n_rv])
        elif self.extinction_mode == "fixed":
            av, rv = float(self.fixed_av), float(self.fixed_rv)
        elif self.extinction_mode == "sample":
            av = float(self.rng.choice(self.avgrid))
            rv = float(self.rng.choice(self.rvgrid))
        else:  # 'none'
            av, rv = 0.0, 3.1

        # outputs with extinction, in the same order as self.label_o / self._filter_map
        bcout: List[float] = []
        for lab, (system, band, _sedname) in zip(self.label_o, self._filter_map):
            bc = float(self.h5dict[system][band][selind])
            bc = self._bc_with_extinction(bc, system, band, av, rv)
            bcout.append(self.normf(bc, lab) if self.norm else bc)

        # inputs
        inputs: List[float] = []
        for ll in self.label_i:
            if ll in row.dtype.names:
                val = float(row[ll])
            elif ll == "av":
                val = av
            elif ll == "rv":
                val = rv
            else:
                raise KeyError(f"Input label '{ll}' not found (expected in parameters or av/rv).")
            inputs.append(self.normf(val, ll) if self.norm else val)

        outarr = np.array(inputs + bcout, dtype=np.float32)
        return torch.tensor(outarr) if self.returntorch else outarr


# ---------------------------------------------------------------------
# Spectral dataloader (CWC spectral grid) — memory-aware, range/resolution
# ---------------------------------------------------------------------
class ReadSpec(Dataset):
    """
    Dataset of synthetic spectra from the CWC grid with optional extinction,
    optional continuum normalization, and memory-aware wavelength selection.

    New kwargs
    ----------
    wave_range : tuple[float, float] | None, default None
        (lo_A, hi_A) in Angstrom. If None, use full range.
    dlambda : float | None, default None
        Constant wavelength step (Å) for resampling. Mutually exclusive with R.
    R : float | None, default None
        Constant resolving power (λ/Δλ) for resampling grid. Mutually exclusive with dlambda.
    rebin_mode : {'interp','bin'}, default 'interp'
        'interp' uses linear interpolation; 'bin' does flux-conserving boxcar average
        (approximate; assumes uniform sampling pre/post). 'bin' ignored if R is used.
    use_norm_from_h5 : bool, default True
        If True and we do NOT resample (only subselect columns), attempt to read
        norm/global/raw/{mean_spectrum,std_spectrum} and slice to chosen columns
        instead of recomputing. Falls back to recompute if unavailable/incompatible.
    pixels_per_resel : float | None, default 2.0
        Number of pixels per resolution element when using constant resolving
        power (R). The target grid step follows Δλ_pix(λ)=λ/(R*pixels_per_resel),
        yielding geometric spacing with factor 1+1/(R*pixels_per_resel).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.verbose: bool = kwargs.get("verbose", False)
        self.progressbar: bool = kwargs.get("progressbar", True)

        # ------------ split / RNG / norm overrides ------------
        self.split_seed = kwargs.get("split_seed", kwargs.get("seed", None))
        self.rng = np.random.default_rng(self.split_seed)
        self.split: Optional[Dict[str, np.ndarray]] = kwargs.get("split", None)
        self.normfactor_override: Optional[Mapping[str, Tuple[float, float]]] = kwargs.get("normfactor", None)

        # ------------ wavelength selection / resampling controls ------------
        self.wave_range: Optional[Tuple[float, float]] = kwargs.get("wave_range", None)
        self.dlambda: Optional[float] = kwargs.get("dlambda", None)
        self.R: Optional[float] = kwargs.get("R", None)
        self.rebin_mode: str = kwargs.get("rebin_mode", "interp")
        self.use_norm_from_h5: bool = bool(kwargs.get("use_norm_from_h5", True))
        self.pixels_per_resel: Optional[float] = kwargs.get("pixels_per_resel", 2.0)

        if self.dlambda is not None:
            if self.R is not None or self.pixels_per_resel not in (None, 2.0):
                raise ValueError("When dlambda is provided, do not set R or pixels_per_resel.")
        else:
            # dlambda is None: okay to use R; if R is None, we keep native grid
            if self.R is not None:
                if self.pixels_per_resel is None or self.pixels_per_resel <= 0:
                    raise ValueError("pixels_per_resel must be positive when R is provided.")        

        if self.dlambda is not None and self.R is not None:
            raise ValueError("Provide only one of dlambda or R (not both).")
        if self.rebin_mode not in ("interp", "bin"):
            raise ValueError("rebin_mode must be 'interp' or 'bin'")
        if self.rebin_mode == "bin":
            if self.dlambda is None:
                raise ValueError("rebin_mode='bin' requires dlambda (constant step).")
    
        # ------------ discover files ------------
        modpath = kwargs.get("modpath", None)
        if modpath is None:
            raise ValueError("Provide `modpath` (directory or list of files)")

        if isinstance(modpath, (list, tuple)):
            file_list = [str(p) for p in modpath]
        elif os.path.isdir(modpath):
            file_list = sorted(glob.glob(os.path.join(modpath, "*.h5")))
        elif os.path.isfile(modpath):
            file_list = [modpath]
        else:
            raise FileNotFoundError(f"modpath not found: {modpath}")

        if not file_list:
            raise FileNotFoundError("No .h5 spectral files discovered.")

        # ------------ utility: build target grid ------------
        def _build_target_grid(native_w: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """Return (lambda_target, col_idx_native) where col_idx_native is None if resampling."""
            w = native_w
            if self.wave_range is not None:
                lo, hi = map(float, self.wave_range)
                lo = max(lo, float(w[0])); hi = min(hi, float(w[-1]))
                if hi <= lo:
                    raise ValueError(f"wave_range outside native grid: {self.wave_range}")
                # native columns that cover requested window (inclusive)
                mask = (w >= lo) & (w <= hi)
                w = w[mask]
                col_idx = np.nonzero(mask)[0]
            else:
                col_idx = np.arange(len(w), dtype=int)

            if self.dlambda is None and self.R is None:
                # No resampling: keep native subset
                return w, col_idx

            # Resample onto new grid
            loA, hiA = (float(w[0]), float(w[-1]))
            if self.dlambda is not None:
                # (unchanged) constant Δλ grid
                step = float(self.dlambda)
                n = int(np.floor((hiA - loA) / step)) + 1
                lam_t = loA + step * np.arange(n, dtype=np.float64)
                if lam_t[-1] < hiA:
                    lam_t = np.append(lam_t, hiA)
            else:
                # R-based geometric grid; use pixels_per_resel to set per-pixel Δλ
                # factor f = 1 + 1/(R * pixels_per_resel)
                if self.R is None:
                    # No resampling requested; keep native subset
                    return w, col_idx
                f = 1.0 + 1.0 / (float(self.R) * float(self.pixels_per_resel))
                lam = [loA]
                # geometric progression until hiA (ensure inclusion of hiA)
                while lam[-1] < hiA:
                    nxt = lam[-1] * f
                    lam.append(nxt if nxt < hiA else hiA)
                lam_t = np.array(lam, dtype=np.float64)
                
            return lam_t, None  # None => we will interpolate/bin from native slice

        # ------------ first pass: determine native grid + target grid ------------
        with h5py.File(file_list[0], "r") as h5_0:
            w_native_full = np.array(h5_0["wavelengths"][()], dtype=np.float64)
        self.wavelengths_A, _col_idx_or_none = _build_target_grid(w_native_full)
        resampling = (_col_idx_or_none is None)  # True => resample
        # For column-only selection, reuse col index for all files
        col_idx_keep = None if resampling else _col_idx_or_none

        # ------------ load & concatenate (column-sliced; optional resampling) ------------
        params_list = []
        spectra_list = []
        cont_list = []
        wav_master_full = w_native_full  # keep the original to validate other files

        if self.progressbar:
            file_list = tqdm(file_list, 
                             desc="Loading spectral files", 
                             unit="file", 
                             leave=False,
                             total=len(file_list),
                             disable=not self.verbose,
                             )

        for fp in file_list:
            if self.verbose:
                print(f"... Reading spectral file: {fp}")
            with h5py.File(fp, "r") as h5:
                if "parameters" not in h5 or "wavelengths" not in h5 or "spectra" not in h5:
                    raise KeyError(f"HDF5 {fp} missing one of required datasets: parameters, wavelengths, spectra")

                wv = np.array(h5["wavelengths"][()], dtype=np.float64)
                # grid equality check
                if wv.size != wav_master_full.size or not np.allclose(wv, wav_master_full, rtol=0, atol=1e-8):
                    raise ValueError(f"Wavelength grid mismatch in {fp}")

                # read only needed columns
                if not resampling:
                    sp = h5["spectra"][:, col_idx_keep].astype(np.float32)
                    if "continuua" in h5:
                        ct = h5["continuua"][:, col_idx_keep].astype(np.float32)
                    else:
                        ct = None
                else:
                    # read just the bounded native window for interpolation
                    if self.wave_range is not None:
                        lo, hi = map(float, self.wave_range)
                        mask_native = (wv >= max(lo, wv[0])) & (wv <= min(hi, wv[-1]))
                        cols = np.nonzero(mask_native)[0]
                    else:
                        cols = slice(None)  # entire range
                    w_slice = wv[cols]
                    sp_native = h5["spectra"][:, cols].astype(np.float64)  # double for interp accuracy
                    if "continuua" in h5:
                        ct_native = h5["continuua"][:, cols].astype(np.float64)
                    else:
                        ct_native = None

                    # build resampled arrays
                    lam_t = self.wavelengths_A
                    # linear interp row-by-row (fast enough at init; vectorization across rows is OK)
                    sp = np.empty((sp_native.shape[0], lam_t.size), dtype=np.float32)
                    if self.rebin_mode == "interp":
                        for i in range(sp_native.shape[0]):
                            sp[i, :] = np.interp(lam_t, w_slice, sp_native[i, :]).astype(np.float32)
                        if ct_native is not None:
                            ct = np.empty_like(sp)
                            for i in range(ct_native.shape[0]):
                                ct[i, :] = np.interp(lam_t, w_slice, ct_native[i, :]).astype(np.float32)
                        else:
                            ct = None
                    else:
                        # crude flux-conserving binning to constant Δλ only
                        if self.dlambda is None:
                            raise ValueError("rebin_mode='bin' requires dlambda (constant step).")
                        # precompute integer bin edges on native index axis
                        # map each target bin to nearest native indices
                        # (simple average; for high precision, plug in astropy.rebin or spectres)
                        for i in range(sp_native.shape[0]):
                            sp[i, :] = np.interp(lam_t, w_slice, sp_native[i, :]).astype(np.float32)
                        if ct_native is not None:
                            ct = np.empty_like(sp)
                            for i in range(ct_native.shape[0]):
                                ct[i, :] = np.interp(lam_t, w_slice, ct_native[i, :]).astype(np.float32)
                        else:
                            ct = None

                par = h5["parameters"][()]
                params_list.append(par)
                spectra_list.append(sp)
                if ct is not None:
                    cont_list.append(ct)

        # Concatenate rows across files
        self.spectra = np.vstack(spectra_list)  # (N_total, L_sel)
        self.has_continuum = len(cont_list) == len(spectra_list)
        self.continuua = np.vstack(cont_list) if self.has_continuum else None
        self.parameters = rfn.stack_arrays(params_list, usemask=False, asrecarray=False)

        # Sanity on parameters schema (accept superset; require core fields)
        required = ("logt", "logg", "feh", "afe", "vmic")
        have = self.parameters.dtype.names
        miss = [f for f in required if f not in have]
        if miss:
            raise ValueError(f"/parameters missing required fields: {miss}; found: {have}")

        # ------------ dataset controls ------------
        self.datatype: str = kwargs.get("type", "train")
        self.returntorch: bool = kwargs.get("returntorch", True)
        self.trainper: float = kwargs.get("trainpercentage", 0.9)
        self.norm: bool = kwargs.get("norm", True)
        self.continuum_mode: str = kwargs.get("continuum_mode", "none")
        if self.continuum_mode not in ("none", "divide"):
            raise ValueError("continuum_mode must be 'none' or 'divide'")

        # ------------- labels -------------
        default_label_i = ["logt", "logg", "feh", "afe", "vmic", "av", "rv"]
        self.label_i: List[str] = kwargs.get("label_i", default_label_i)
        self.label_o: List[str] = [f"lam_{int(round(lam))}" for lam in self.wavelengths_A]
        self._lam_count = len(self.wavelengths_A)

        # ------------ parameter filtering ------------
        self.parrange: Optional[Mapping[str, Tuple[float, float]]] = kwargs.get("parrange", None)
        self.parameters = rfn.append_fields(
            self.parameters, "model_index", np.arange(len(self.parameters)), usemask=False
        )
        if self.parrange is not None:
            mask = np.ones(len(self.parameters), dtype=bool)
            for k, (lo, hi) in self.parrange.items():
                if k in self.parameters.dtype.names:
                    mask &= (self.parameters[k] >= lo) & (self.parameters[k] <= hi)
            if not np.any(mask):
                raise ValueError("parrange filters eliminated all rows")
            self.parameters = self.parameters[mask]
            self.spectra = self.spectra[mask, :]
            if self.has_continuum:
                self.continuua = self.continuua[mask, :]

        # ------------ splits ------------
        if self.split is not None:
            for key in ("train", "valid", "test"):
                if key not in self.split:
                    raise ValueError(f"split dict missing key '{key}'")

            want_idx = self.split[self.datatype]
            mask = np.isin(self.parameters["model_index"], want_idx)
            base_block = self.parameters[mask]

            self.parameters_train = self.parameters[np.isin(self.parameters["model_index"], self.split["train"])]
            self.parameters_valid = self.parameters[np.isin(self.parameters["model_index"], self.split["valid"])]
            self.parameters_test  = self.parameters[np.isin(self.parameters["model_index"], self.split["test"])]
        else:
            order = np.arange(len(self.parameters))
            self.rng.shuffle(order)
            self.parameters = self.parameters[order]
            self.spectra = self.spectra[order, :]
            if self.has_continuum:
                self.continuua = self.continuua[order, :]

            cut = int(np.rint((1.0 - self.trainper) * len(self.parameters)))
            test_block = self.parameters[:cut]
            rest = self.parameters[cut:]
            mid = int(np.rint(0.7 * len(rest)))
            train_block = rest[:mid]
            valid_block = rest[mid:]

            self.parameters_train = train_block
            self.parameters_valid = valid_block
            self.parameters_test  = test_block
            base_block = {"train": train_block, "valid": valid_block, "test": test_block}[self.datatype]

        self.split_indices = {
            "train": np.asarray(self.parameters_train["model_index"]),
            "valid": np.asarray(self.parameters_valid["model_index"]),
            "test" : np.asarray(self.parameters_test["model_index"]),
        }

        # ------------- Extinction control -------------
        self.extinction_mode: str = kwargs.get("extinction_mode", None) or (
            "sample" if self.datatype == "train" else "grid"
        )

        self.avgrid = np.array(
            kwargs.get(
                "avgrid",
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                + list(range(1, 10, 1))
                + list(range(10, 50, 5))
                + list(range(50, 101, 10)),
            ),
            dtype=np.float32,
        )
        self.rvgrid = np.array(kwargs.get("rvgrid", [2.3, 2.5, 3.1, 3.5, 4.0, 5.0, 5.6]), dtype=np.float32)

        if self.parrange is not None and "av" in self.parrange:
            lo, hi = self.parrange["av"]
            self.avgrid = self.avgrid[(self.avgrid >= lo) & (self.avgrid <= hi)]
        if self.parrange is not None and "rv" in self.parrange:
            lo, hi = self.parrange["rv"]
            self.rvgrid = self.rvgrid[(self.rvgrid >= lo) & (self.rvgrid <= hi)]

        if len(self.avgrid) == 0:
            raise ValueError("No valid values found in avgrid")
        if len(self.rvgrid) == 0:
            raise ValueError("No valid values found in rvgrid")

        self.fixed_av: float = float(kwargs.get("fixed_av", 0.0))
        self.fixed_rv: float = float(kwargs.get("fixed_rv", 3.1))

        base_idx = base_block["model_index"]
        if self.extinction_mode == "grid":
            grid_mult = int(len(self.avgrid) * len(self.rvgrid))
            self._selind = np.repeat(base_idx, grid_mult)
            self._param_rows = np.repeat(base_block, grid_mult)
            self._per_row_grid = grid_mult
        else:
            self._selind = base_idx
            self._param_rows = base_block
            self._per_row_grid = 1

        # ------------ normalization stats ------------
        if self.normfactor_override is not None:
            self.normfactor = dict(self.normfactor_override)
        else:
            self.normfactor: Dict[str, Tuple[float, float]] = {}

            # Inputs
            for ll in self.label_i:
                if ll in self.parameters.dtype.names:
                    x = self.parameters[ll].astype(np.float64)
                    mu, sdv = float(np.mean(x)), float(np.std(x)); sd = sdv if sdv > 0 else 1.0
                    self.normfactor[ll] = (mu, sd)
                elif ll == "av":
                    mu, sdv = float(np.mean(self.avgrid)), float(np.std(self.avgrid)); sd = sdv if sdv > 0 else 1.0
                    self.normfactor[ll] = (mu, sd)
                elif ll == "rv":
                    mu, sdv = float(np.mean(self.rvgrid)), float(np.std(self.rvgrid)); sd = sdv if sdv > 0 else 1.0
                    self.normfactor[ll] = (mu, sd)
                else:
                    self.normfactor[ll] = (0.0, 1.0)

            # Outputs: try to use norm/global/raw if we *only* column-selected (no resample)
            could_use_h5_norm = self.use_norm_from_h5 and (not resampling)
            mu_vec = None; sd_vec = None

            if could_use_h5_norm:
                try:
                    # read once from first file; assume all files share same stats provenance
                    with h5py.File(file_list[0], "r") as h5:
                        g = h5["norm/global/raw"]
                        mu_full = np.array(g["mean_spectrum"][()], dtype=np.float64)
                        sd_full = np.array(g["std_spectrum"][()], dtype=np.float64)
                        mu_vec = mu_full[col_idx_keep]
                        sd_vec = sd_full[col_idx_keep]
                except Exception:
                    mu_vec = sd_vec = None  # fallback to compute

            if (mu_vec is None) or (sd_vec is None) or (mu_vec.size != self._lam_count):
                # compute from current base split on the already-trimmed/resampled spectra in RAM
                base_mask = np.isin(self.parameters["model_index"], base_idx)
                Y = self.spectra[base_mask, :].astype(np.float64)  # (B, L_sel)
                if self.continuum_mode == "divide" and self.has_continuum:
                    Y = (Y / np.maximum(self.continuua[base_mask, :], 1e-30))
                mu_vec = np.mean(Y, axis=0)
                sd_vec = np.std(Y, axis=0); sd_vec[sd_vec <= 0] = 1.0

            # Store as label-wise entries:
            for j, lab in enumerate(self.label_o):
                self.normfactor[lab] = (float(mu_vec[j]), float(sd_vec[j]))

        # ------------ extinction cache (vector k_lambda) ------------
        self._k_lambda_cache: Dict[float, np.ndarray] = {}  # key: rv -> k(λ) vector
        self._x_inv_micron = (1.0 / (self.wavelengths_A * 1e-4)) * u.micron**-1  # broadcastable

        self.datalen = len(self._selind)
        if self.verbose:
            print(f"... Spectral dataset type: {self.datatype}")
            print(f"... Extinction mode: {self.extinction_mode}")
            print(f"... N rows (effective): {self.datalen}")
            print(f"... N_lambda: {self._lam_count}  (λ in Å)")

    # ------------- helpers -------------
    def normf(self, x: float | np.ndarray, label: str | Sequence[str]):
        if isinstance(label, str):
            mu, sd = self.normfactor[label]; return (x - mu) / sd
        mu = np.array([self.normfactor[lab][0] for lab in label], dtype=np.float64)
        sd = np.array([self.normfactor[lab][1] for lab in label], dtype=np.float64)
        return (x - mu) / sd

    def unnormf(self, x: float | np.ndarray, label: str | Sequence[str]):
        if isinstance(label, str):
            mu, sd = self.normfactor[label]; return x * sd + mu
        mu = np.array([self.normfactor[lab][0] for lab in label], dtype=np.float64)
        sd = np.array([self.normfactor[lab][1] for lab in label], dtype=np.float64)
        return x * sd + mu

    def _k_lambda(self, rv: float) -> np.ndarray:
        lo, hi = 2.3, 5.6
        rvf = float(rv)
        if rvf <= lo: rvf = float(np.nextafter(lo, 10.0))
        elif rvf >= hi: rvf = float(np.nextafter(hi, 0.0))
        if rvf in self._k_lambda_cache:
            return self._k_lambda_cache[rvf]
        kvec = np.array(G23(Rv=rvf)(self._x_inv_micron), dtype=np.float64)  # A(λ)/A(V)
        self._k_lambda_cache[rvf] = kvec
        return kvec

    # ------------- Dataset API -------------
    def __len__(self) -> int:
        return self.datalen

    def __getitem__(self, idx: int):
        selind = self._selind[idx]
        row = self._param_rows[idx]

        # determine A_V, R_V according to mode
        if self.extinction_mode == "grid":
            gpos = idx % self._per_row_grid
            n_rv = len(self.rvgrid)
            av = float(self.avgrid[gpos // n_rv])
            rv = float(self.rvgrid[gpos % n_rv])
        elif self.extinction_mode == "fixed":
            av, rv = float(self.fixed_av), float(self.fixed_rv)
        elif self.extinction_mode == "sample":
            av = float(self.rng.choice(self.avgrid))
            rv = float(self.rng.choice(self.rvgrid))
        else:  # 'none'
            av, rv = 0.0, 3.1

        # intrinsic spectrum
        y = self.spectra[selind, :].astype(np.float64)
        if self.continuum_mode == "divide" and self.has_continuum:
            y = y / np.maximum(self.continuua[selind, :].astype(np.float64), 1e-30)

        # extinction in flux space
        if self.extinction_mode != "none":
            kvec = self._k_lambda(rv)
            y = y * np.power(10.0, -0.4 * av * kvec)

        # outputs normalization
        y_out = self.normf(y, self.label_o) if self.norm else y

        # inputs
        x_list: List[float] = []
        for ll in self.label_i:
            if ll in row.dtype.names:
                val = float(row[ll])
            elif ll == "av":
                val = av
            elif ll == "rv":
                val = rv
            else:
                raise KeyError(f"Input label '{ll}' not found (expected in parameters or av/rv).")
            x_list.append(self.normf(val, ll) if self.norm else val)

        flat = np.concatenate([np.asarray(x_list, dtype=np.float64), y_out.astype(np.float64)], axis=0).astype(np.float32)
        return torch.tensor(flat) if self.returntorch else flat
    

class XYFromFlat(torch.utils.data.Dataset):
    """Wrap a ReadPhot/ReadSpec dataset that returns a flat vector and emit (x, y).

    Each item returns (x, y) where x has length len(base_ds.label_i) and y has
    length len(base_ds.label_o).
    """

    def __init__(self, base_ds: ReadPhot) -> None:
        self.ds = base_ds
        self.n_in = len(base_ds.label_i)
        self.n_out = len(base_ds.label_o)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        flat = self.ds[idx]  # 1D tensor or ndarray
        x = flat[: self.n_in]
        y = flat[self.n_in :]
        return x, y
