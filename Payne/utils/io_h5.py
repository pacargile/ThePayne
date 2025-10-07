import h5py
import numpy as np
import torch

def _ensure_group(f: h5py.File, path: str):
    if path in f:
        del f[path]
    return f.create_group(path)

def save_state_dict_to_h5(state_dict: dict, h5path: str, group: str = "model", compression="gzip"):
    """
    Save a torch state_dict to HDF5 under /<group>/<param_name>.
    Arrays are stored as float32/float64/int64 as given by the tensors.
    """
    with h5py.File(h5path, "a") as f:
        g = _ensure_group(f, group)
        for k, v in state_dict.items():
            arr = v.detach().cpu().numpy()
            g.create_dataset(k, data=arr, compression=compression)

def load_state_dict_from_h5(
    model: torch.nn.Module,
    h5path: str,
    group: str = "model",
    strict: bool = True,
    dtype=None
):
    """
    Load parameters from an HDF5 checkpoint into a model.

    Automatically handles torch.compile()-wrapped models
    (which prefix keys with '_orig_mod.') and optionally
    casts tensors to a specified dtype (e.g., torch.float32).
    """
    import h5py, torch

    sd = {}
    with h5py.File(h5path, "r") as f:
        if group not in f:
            raise KeyError(f"Group '{group}' not found in {h5path}")
        g = f[group]
        for k in g.keys():
            arr = g[k][()]
            t = torch.from_numpy(arr)
            if dtype is not None:
                t = t.to(dtype=dtype)
            sd[k] = t

    # --- Handle compile wrapper / prefixed keys ---
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    # Load into underlying module if wrapped
    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(sd, strict=strict)

    return model

def save_labels_norms_to_h5(h5path: str, label_i, label_o, normfactor=None):
    """
    Save label lists and (optionally) normalization (mean,std) per label.
    Strings are stored as ASCII bytes for max compatibility.
    """
    to_bytes = lambda xs: np.array([str(x).encode("ascii", "ignore") for x in xs])
    with h5py.File(h5path, "a") as f:
        # labels
        if "label_i" in f: del f["label_i"]
        if "label_o" in f: del f["label_o"]
        f.create_dataset("label_i", data=to_bytes(label_i))
        f.create_dataset("label_o", data=to_bytes(label_o))

        # norms
        if normfactor is not None:
            # clear old groups if exist
            if "norm_i" in f: del f["norm_i"]
            if "norm_o" in f: del f["norm_o"]
            gi = f.create_group("norm_i")
            go = f.create_group("norm_o")
            for k in label_i:
                if k in normfactor:
                    gi.create_dataset(k, data=np.array(normfactor[k], dtype=np.float64))
            for k in label_o:
                if k in normfactor:
                    go.create_dataset(k, data=np.array(normfactor[k], dtype=np.float64))

def save_meta_to_h5(h5path: str, **meta):
    with h5py.File(h5path, "a") as f:
        g = f.get("meta", None)
        if g is None:
            g = f.create_group("meta")
        for k, v in meta.items():
            # store simple scalars/strings as attrs
            try:
                g.attrs[k] = v
            except TypeError:
                g.attrs[k] = str(v)