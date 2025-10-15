import os, glob, h5py, re
import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from torch.utils.data import Dataset
from dust_extinction.parameter_averages import G23
from astropy import units as u

try:
    from sedpy import observate
    from sedpy.observate import Filter as SEDpyFilter
except Exception as e:
    raise ImportError("`sedpy` is required when using `filters`.") from e

# ------------------------------
# Inlined SEDpy wavelength tools
# ------------------------------
def _pivot_wavelength(wave_A, trans):
    w = np.asarray(wave_A, dtype=float)
    S = np.asarray(trans, dtype=float)
    num = np.trapz(S * w, w)
    den = np.trapz(S / w, w)
    return np.sqrt(num / den)

def _logmean_wavelength(wave_A, trans):
    w = np.asarray(wave_A, dtype=float); S = np.asarray(trans, dtype=float)
    lnw = np.log(w); dlnw = np.gradient(lnw)
    return np.exp(np.sum(lnw * S * dlnw) / np.sum(S * dlnw))

def _as_dict_of_paths(modpath):
    if isinstance(modpath, dict): return modpath
    if os.path.isdir(modpath):
        out = {}
        for p in sorted(glob.glob(os.path.join(modpath, "*.h5"))):
            sysname = os.path.splitext(os.path.basename(p))[0]
            out[sysname] = p
        if not out: raise FileNotFoundError(f"No .h5 under: {modpath}")
        return out
    if os.path.isfile(modpath):
        return {"__single__": modpath}
    raise FileNotFoundError(f"modpath not found: {modpath}")

def _first_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            # skip callables; accept arrays/lists
            if v is not None and not callable(v):
                return v
    return None

def _get_filter_arrays(f):
    """
    Return (wave_A, trans) from a sedpy Filter object, robust to API variants.
    wave_A in Angstrom, trans unitless.
    """
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
_DEFAULT_PREFIX_MAP = {
    r"^ps_":         ("panstarrs", lambda name: name.split("_", 1)[1]),     # panstarrs_g -> g
    r"^gaia_":       ("gaia",      lambda name: name.split("_", 1)[1]),     # gaia_bp -> bp
    r"^twomass_":    ("twomass",   lambda name: name.split("_", 1)[1]),     # twomass_J -> J
    r"^wise_":       ("wise",      lambda name: name.split("_", 1)[1]),     # wise_w1 -> w1
    r"^sdss_":       ("sdss", lambda name: name.split("_", 1)[1].rstrip("0")),     # sdss_g0 -> g
    r"^decam_":      ("decam",     lambda name: name.split("_", 1)[1]),     # decam_g -> g
    r"^lsst_":       ("lsst",      lambda name: name.split("_", 1)[1]),     # lsst_r -> r
    r"^roman_":      ("roman_wfi", lambda name: name.split("_", 2)[2]),     # roman_wfi_f062 -> f062
    r"^swift_":      ("uvot",      lambda name: name.split("_", 1)[1]),     # uvot_f062 -> f062
    r"^spx_":        ("spherex",   lambda name: name.split("_", 1)[1]),     # spherex_ch062 -> ch062
}

def _resolve_system_band_from_sedpy_name(
    sedpy_name,
    h5_system_names,
    h5_fields_by_system,
    user_system_alias=None,
    user_band_alias=None,
):
    """
    Map a SEDpy filter name to (system, band) used in your HDF5.

    - Uses _DEFAULT_PREFIX_MAP (incl. Roman WFI).
    - Applies optional user aliases.
    - Adapts band 'case' automatically to match HDF5 field names.
    """
    name_lc = sedpy_name.lower()

    # 1) Rule-based inference of system + band guess
    system = None; band_guess = None
    for pref, (sysname, band_fn) in _DEFAULT_PREFIX_MAP.items():
        if re.match(pref, name_lc):
            system = sysname
            band_guess = band_fn(sedpy_name)  # may be lower/upper/mixed
            break
    if system is None:
        # fallback: split at first underscore
        parts = sedpy_name.split("_", 1)
        system = parts[0].lower() if len(parts) == 2 else sedpy_name.lower()
        band_guess = parts[1] if len(parts) == 2 else sedpy_name

    # 2) Apply user aliases (optional)
    if user_system_alias and system in user_system_alias:
        system = user_system_alias[system]
    if user_band_alias and (system, band_guess) in user_band_alias:
        band_guess = user_band_alias[(system, band_guess)]

    # 3) Snap system to a real HDF5 system name if needed
    if system not in h5_system_names:
        # try prefix match (e.g., "roman" → "roman_wfi") or case-insensitive match
        cand = [s for s in h5_system_names if s.lower() == system.lower() or s.lower().startswith(system)]
        if len(cand) == 1:
            system = cand[0]
        elif len(cand) > 1:
            # pick the longest (most specific) match
            system = sorted(cand, key=len, reverse=True)[0]
        else:
            raise KeyError(f"System '{system}' (from '{sedpy_name}') not found in HDF5 systems {sorted(h5_system_names)}")

    # 4) Snap band to an exact HDF5 field (case-insensitive, tolerant to small style diffs)
    fields = list(h5_fields_by_system.get(system, []))
    if not fields:
        raise KeyError(f"No filter fields found for system '{system}' in HDF5.")

    # exact match first
    if band_guess in fields:
        return system, band_guess

    # build normalization maps for tolerant matching
    def _norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    fields_norm = {_norm(f): f for f in fields}       # normalized field → original field
    band_norm = _norm(band_guess)

    # try (i) case-insensitive exact, (ii) normalized matching
    ci_map = {f.lower(): f for f in fields}
    if band_guess.lower() in ci_map:
        return system, ci_map[band_guess.lower()]
    if band_norm in fields_norm:
        return system, fields_norm[band_norm]

    # last resort: prefix/contains match on normalized keys
    hits = [orig for norm, orig in fields_norm.items() if norm.startswith(band_norm) or band_norm.startswith(norm)]
    if len(hits) == 1:
        return system, hits[0]

    raise KeyError(
        f"Cannot map SEDpy filter '{sedpy_name}' to HDF5 ({system}, '{band_guess}'). "
        f"Available fields in '{system}': {sorted(fields)}"
    )
    
class ReadPhot(Dataset):
    """
    Read BC tables (no Av/Rv baked in) and apply G23 extinction on-the-fly.

    Pass a simple list of SEDpy filter names via `filters=[...]`.
    The loader will:
      * compute representative wavelengths with SEDpy (pivot by default),
      * resolve each SEDpy name to (system, band) in your HDF5,
      * build outputs only for those requested bands.

    HDF5 layout (per-system file OR legacy single file):
      - 'parameters': structured array with fields ('logt','logg','feh','afe','vmic')
      - '<system>': structured array with one field per filter (e.g., 'g','r','i',...)

    Key kwargs:
      modpath: str|dict
      filters: list[str]  # SEDpy names, e.g. ['ps1_g','ps1_r','gaia_g','gaia_bp',...]
      filter_wavelength_method: 'pivot'|'logmean' (default 'pivot')
      system_alias: dict[str,str] optional renames for systems in HDF5
      band_alias: dict[(system,band_inferred)->band_in_h5] optional band remaps

      type: 'train'|'valid'|'test'
      extinction_mode: 'sample' (default train) | 'grid' (default valid/test) | 'fixed' | 'none'
      avgrid, rvgrid, fixed_av, fixed_rv
      norm: z-score inputs/outputs (outputs normed on intrinsic BCs)

      label_i default: ['logt','logg','feh','afe','av','rv']
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.verbose = kwargs.get('verbose', False)

        # --- reproducible splits + norm override ---
        self.split_seed = kwargs.get('split_seed', kwargs.get('seed', None))
        self.rng = np.random.default_rng(self.split_seed)

        # Externally supplied split indices (by model_index values)
        # Example: split={'train': np.array([...]), 'valid': np.array([...]), 'test': np.array([...])}
        self.split = kwargs.get('split', None)

        # Externally supplied normalization dict: {label: (mean, std)}
        # Must cover all labels in label_i + label_o if provided.
        self.normfactor_override = kwargs.get('normfactor', None)
        
        # ---------------- Sources ----------------
        self.modpaths = _as_dict_of_paths(kwargs.get('modpath', None))
        if self.modpaths is None:
            raise ValueError("Provide modpath")

        # HDF5 file paths
        self.h5dict = {}
        self._systems = []

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
                # schema check
                expected = ("logt","logg","feh","afe","vmic")
                have = self.parameters.dtype.names
                missing = [f for f in expected if f not in have]
                if missing:
                    raise ValueError(f"/parameters missing fields: {missing}; found: {have}")

                # ----- optional meta & rowkey -----
                if "meta" in h5:
                    # meta is a group or dataset with attrs; capture attrs only
                    self.meta = dict(h5["meta"].attrs)
                else:
                    self.meta = {}

                if "rowkey" in h5 and isinstance(h5["rowkey"], h5py.Dataset):
                    self.rowkey = h5["rowkey"][()]
                    # optional sanity: same length as parameters
                    if len(self.rowkey) != len(self.parameters):
                        raise ValueError(f"/rowkey length {len(self.rowkey)} != /parameters length {len(self.parameters)}")
                else:
                    self.rowkey = None

                # ----- which photometric systems to load? -----
                # Determine requested systems from the provided filters using your prefix map.
                sedpy_filters = kwargs.get("filters", None)
                if not sedpy_filters:
                    raise ValueError("Pass `filters` (list of filter names).")

                requested_systems = set()
                for fname in sedpy_filters:
                    # use your resolver/prefix map to get (system, _band_guess)
                    # if you have a helper, call it; otherwise a minimal fallback:
                    name_lc = fname.lower()
                    matched = False
                    for pref, (sysname, band_fn) in _DEFAULT_PREFIX_MAP.items():
                        if re.match(pref, name_lc):
                            requested_systems.add(sysname)
                            matched = True
                            break
                    if not matched:
                        # fallback: assume <system>_<band>
                        parts = fname.split("_", 1)
                        if len(parts) == 2:
                            requested_systems.add(parts[0].lower())

                # Available dataset names at root (excluding non-systems)
                NON_SYSTEM_KEYS = {"parameters", "meta", "rowkey"}
                available_systems = {k for k, v in h5.items()
                                    if isinstance(v, h5py.Dataset) and k not in NON_SYSTEM_KEYS}
                # Guard: every requested system must exist
                missing_sys = sorted(requested_systems - available_systems)
                if missing_sys:
                    raise KeyError(f"Requested systems {missing_sys} not found. "
                                f"Available systems in file: {sorted(available_systems)}")

                # ----- load only the requested system datasets -----
                for sysname in sorted(requested_systems):
                    ds = h5[sysname]
                    if ds.dtype.names is None:
                        raise TypeError(f"Dataset '/{sysname}' must be a structured array with one field per band.")
                    self.h5dict[sysname] = ds[()]   # load structured array
                    self._systems.append(sysname)
        else:
            params_ref = None
            for sysname, path in self.modpaths.items():
                if self.verbose:
                    print(f"... Reading {sysname} from {path}")
                with h5py.File(path, "r") as h5:
                    params = h5["parameters"][()]
                    phot   = h5[sysname][()]
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
        fields_by_system = {sys: list(self.h5dict[sys].dtype.names) for sys in self._systems}

        # ----------- SEDpy filters list → wavelengths + mapping -----------
        sedpy_filters = kwargs.get('filters', None)
        if not sedpy_filters:
            raise ValueError("Pass `filters` as a list of SEDpy filter names (e.g., ['ps1_g','gaia_g', ...])")

        method = kwargs.get('filter_wavelength_method', 'pivot')
        system_alias = kwargs.get('system_alias', None)
        band_alias   = kwargs.get('band_alias', None)

        # Load SEDpy filters
        sed_list = list(sedpy_filters)

        sed_objs = []
        for ff in sed_list:
            if ff.lower().startswith("spherex"):  # treat anything starting with spherex
                sed_objs.append(SEDpyFilter(kname="spherex", trans_colname=ff.split('_')[1]))
            else:
                # observate.load_filters returns a list, but we want a single Filter object here
                sed_objs.extend(observate.load_filters([ff]))

        sed_by_name = {f.name: f for f in sed_objs}

        # Compute λ and map to (system, band) in HDF5
        self.filter_wavelengths = {}           # {system:{band: lambda_A}}
        self._out_labels = []                  # ["system_band", ...] in the same order as `filters`
        self._filter_map = []                  # list of tuples for fast access: (system, band, sedpy_name)

        for sname in sed_list:
            f = sed_by_name[sname]
            wA, T = _get_filter_arrays(f)
            lamA = float(_pivot_wavelength(wA, T) if method == "pivot"
                        else _logmean_wavelength(wA, T))
            # lazily fill system dict
            # we must map sname -> (system, band) in HDF5
            system, band = _resolve_system_band_from_sedpy_name(
                sname, set(self._systems), fields_by_system,
                user_system_alias=system_alias, user_band_alias=band_alias
            )
            if system not in self.filter_wavelengths:
                self.filter_wavelengths[system] = {}
            self.filter_wavelengths[system][band] = lamA
            self._out_labels.append(f"{system}_{band}")
            self._filter_map.append((system, band, sname))

        # ------------- Dataset type / splits -------------
        self.datatype = kwargs.get('type', 'train')
        self.returntorch = kwargs.get('returntorch', True)
        self.trainper = kwargs.get('trainpercentage', 0.9)
        self.norm = kwargs.get('norm', True)

        # ------------- Labels -------------
        default_label_i = ['logt','logg','feh','afe','av','rv']
        self.label_i = kwargs.get('label_i', default_label_i)
        self.label_o = kwargs.get('label_o', self._out_labels)

        # check to make sure norm factors are provided
        if self.normfactor_override is not None:
            missing = [k for k in (self.label_i + self.label_o) if k not in self.normfactor_override]
            if missing:
                raise ValueError(f"normfactor override missing labels: {missing}")
    

        # ------------- Parameter selection -------------
        self.parrange = kwargs.get('parrange', None)
        self.parameters = rfn.append_fields(self.parameters, 'model_index',
                                            np.arange(len(self.parameters)), usemask=False)
        if self.parrange is not None:
            for k, (lo, hi) in self.parrange.items():
                if self.verbose: print(f"... Applying parameter range for {k}: [{lo},{hi}]")
                try:
                    self.parameters = self.parameters[(self.parameters[k] >= lo) & (self.parameters[k] <= hi)]
                except ValueError: # catch Av and Rv because these aren't defined yet
                    pass

        # --- build splits ---
        # If explicit split indices were supplied, use them.
        # Otherwise, do a deterministic shuffle with split_seed and then slice.

        # check if splits are correctly supplied
        if self.split is not None:
            for key in ("train","valid","test"):
                if key not in self.split:
                    raise ValueError(f"split dict missing key '{key}'")

        if self.split is not None:
            # Keep only rows present in the supplied split for this datatype
            want = self.split.get(self.datatype, None)
            if want is None:
                raise ValueError(f"split dict missing key '{self.datatype}'")
            # map from model_index → row boolean
            mask = np.isin(self.parameters['model_index'], want)
            base_block = self.parameters[mask]
            # also keep the other splits for export
            self.parameters_train = self.parameters[np.isin(self.parameters['model_index'], self.split.get('train', []))]
            self.parameters_valid = self.parameters[np.isin(self.parameters['model_index'], self.split.get('valid', []))]
            self.parameters_test  = self.parameters[np.isin(self.parameters['model_index'], self.split.get('test',  []))]

        else:
            # Deterministic shuffle using split_seed (or None → nondeterministic)
            self.rng.shuffle(self.parameters)

            cut = int(np.rint((1.0 - self.trainper) * len(self.parameters)))
            test_block = self.parameters[:cut]
            rest = self.parameters[cut:]
            mid = int(np.rint(0.7 * len(rest)))

            train_block = rest[:mid]
            valid_block = rest[mid:]

            self.parameters_train = train_block
            self.parameters_valid = valid_block
            self.parameters_test  = test_block

            base_block = {'train': train_block, 'valid': valid_block, 'test': test_block}[self.datatype]
    
        # keep handy for saving out
        self.split_indices = {
            'train': np.asarray(self.parameters_train['model_index']) if hasattr(self, 'parameters_train') else np.array([], dtype=int),
            'valid': np.asarray(self.parameters_valid['model_index']) if hasattr(self, 'parameters_valid') else np.array([], dtype=int),
            'test':  np.asarray(self.parameters_test['model_index'])  if hasattr(self, 'parameters_test')  else np.array([], dtype=int),
        }

        # ------------- Extinction control -------------
        self.extinction_mode = kwargs.get('extinction_mode', None)
        if self.extinction_mode is None:
            self.extinction_mode = 'sample' if self.datatype == 'train' else 'grid'

        self.avgrid = kwargs.get('avgrid',
            [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] +
            list(range(1,10,1)) + list(range(10,50,5)) + list(range(50,101,10))
        )
        self.rvgrid = kwargs.get('rvgrid', [2.3,2.5,3.1,3.5,4.0,5.0,5.6])

        # parse avgrid and rvgrid based on input parrange
        self.avgrid = np.array(self.avgrid, dtype=np.float32)
        self.rvgrid = np.array(self.rvgrid, dtype=np.float32)
        if self.parrange is not None and 'av' in self.parrange:
            self.avgrid = self.avgrid[(self.avgrid >= min(self.parrange['av'])) & (self.avgrid <= max(self.parrange['av']))]
        if self.parrange is not None and 'rv' in self.parrange:
            self.rvgrid = self.rvgrid[(self.rvgrid >= min(self.parrange['rv'])) & (self.rvgrid <= max(self.parrange['rv']))]

        # check to make sure we have something in the grids
        if len(self.avgrid) == 0:
            raise ValueError("No valid values found in avgrid")
        if len(self.rvgrid) == 0:
            raise ValueError("No valid values found in rvgrid")

        self.fixed_av = kwargs.get('fixed_av', 0.0)
        self.fixed_rv = kwargs.get('fixed_rv', 3.1)

        base_idx = base_block['model_index']
        if self.extinction_mode == 'grid':
            self._av_list = np.array([a for a in self.avgrid for _ in self.rvgrid], dtype=np.float32)
            self._rv_list = np.array([r for _ in self.avgrid for r in self.rvgrid], dtype=np.float32)
            self._selind = np.repeat(base_idx, len(self._av_list))
            self._param_rows = np.repeat(base_block, len(self._av_list))
        else:
            self._selind = base_idx; self._param_rows = base_block

        # ------------- Normalization (intrinsic BCs) -------------
        if self.normfactor_override is not None:
            # Use supplied (mean,std) for all labels
            self.normfactor = dict(self.normfactor_override)
        else:
            self.normfactor = {}
            # Inputs
            for ll in self.label_i:
                if ll in base_block.dtype.names:
                    x = base_block[ll].astype(np.float64)
                elif ll == 'av':
                    x = np.array(self.avgrid, dtype=np.float64)
                elif ll == 'rv':
                    x = np.array(self.rvgrid, dtype=np.float64)
                else:
                    self.normfactor[ll] = (0.0, 1.0); continue
                mu = float(np.mean(x))
                sd = float(np.std(x)) if np.std(x) > 0 else 1.0
                self.normfactor[ll] = (mu, sd)

            # Outputs (intrinsic, unreddened) – stats computed over *all* rows to be robust
            for lab, (system, band, _) in zip(self.label_o, self._filter_map):
                bc_arr = self.h5dict[system][band].astype(np.float64)
                mu = float(np.mean(bc_arr))
                sd = float(np.std(bc_arr)) if np.std(bc_arr) > 0 else 1.0
                self.normfactor[lab] = (mu, sd)

        # ------------- k(λ) cache -------------
        self._k_cache = {}  # (rv, system, band) -> k_lambda
        self.datalen = len(self._selind)

        if self.verbose:
            print(f"... Data Set Type: {self.datatype}")
            print(f"... Extinction mode: {self.extinction_mode}")
            print(f"... N rows (effective): {self.datalen}")
            print(f"... Systems present: {self._systems}")
            print(f"... Outputs: {self.label_o}")

    # ------------- helpers -------------
    def normf(self, x, label):
        mu, sd = self.normfactor[label]; return (x - mu) / sd
    def unnormf(self, x, label):
        mu, sd = self.normfactor[label]; return x * sd + mu

    def _k_for(self, rv, system, band):
        key = (float(rv), system, band)
        if key in self._k_cache:
            return self._k_cache[key]

        lamA = self.filter_wavelengths[system][band]    # Å
        # --- clamp Rv to the valid open interval (avoid FP boundary hits)
        lo, hi = 2.3, 5.6
        rvf = float(rv)
        if rvf <= lo:
            rvf = float(np.nextafter(lo, 10.0))     # smallest float > 2.3
        elif rvf >= hi:
            rvf = float(np.nextafter(hi, 0.0))      # largest float < 5.6

        # use units path (silences warning and is explicit)
        from astropy import units as u
        x_inv_micron = (1.0 / (lamA * 1e-4)) * u.micron**-1
        k = float(G23(Rv=rvf)(x_inv_micron))  # A(λ)/A(V)

        self._k_cache[key] = k
        return k

    def _bc_with_extinction(self, bc_intrinsic, system, band, av, rv):
        if self.extinction_mode == 'none': return bc_intrinsic
        return bc_intrinsic - av * self._k_for(rv, system, band)

    # ------------- Dataset API -------------
    def __len__(self): return self.datalen

    def __getitem__(self, idx):
        selind = self._selind[idx]
        row = self._param_rows[idx]

        if self.extinction_mode == 'grid':
            per_row = len(self.avgrid) * len(self.rvgrid)
            gpos = idx % per_row
            av = float(self.avgrid[gpos // len(self.rvgrid)])
            rv = float(self.rvgrid[gpos % len(self.rvgrid)])
        elif self.extinction_mode == 'fixed':
            av, rv = float(self.fixed_av), float(self.fixed_rv)
        elif self.extinction_mode == 'sample':
            av = float(self.rng.choice(self.avgrid)); rv = float(self.rng.choice(self.rvgrid))
        else:
            av, rv = 0.0, 3.1

        # outputs with extinction, in the same order as self.label_o / self._filter_map
        bcout = []
        for lab, (system, band, _) in zip(self.label_o, self._filter_map):
            bc = self.h5dict[system][band][selind]
            bc = self._bc_with_extinction(bc, system, band, av, rv)
            if self.norm: bc = self.normf(bc, lab)
            bcout.append(bc)

        # inputs
        inputs = []
        for ll in self.label_i:
            if ll in row.dtype.names: val = float(row[ll])
            elif ll == 'av':         val = av
            elif ll == 'rv':         val = rv
            else: raise KeyError(f"Input label '{ll}' not found (expected in parameters or av/rv).")
            inputs.append(self.normf(val, ll) if self.norm else val)

        outarr = np.array(inputs + bcout, dtype=np.float32)
        return torch.tensor(outarr) if self.returntorch else outarr
    
class XYFromFlat(torch.utils.data.Dataset):
    """
    Wrap a ReadPhot dataset (which returns 1D tensor [n_in + n_out])
    and return (x, y) tensors directly.
    """
    def __init__(self, base_ds):
        self.ds = base_ds
        self.n_in = len(base_ds.label_i)
        self.n_out = len(base_ds.label_o)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        flat = self.ds[idx]  # 1D tensor
        x = flat[: self.n_in]
        y = flat[self.n_in :]
        return x, y
