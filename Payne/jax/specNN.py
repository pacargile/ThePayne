import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import nnx

import warnings
import h5py
import time,sys,os,glob
from datetime import datetime

def _exists(h5, path: str) -> bool:
    try:
        h5[path]
        return True
    except Exception:
        return False

def _read_norms(nnh5, label_i, label_o):
    """
    Return norm_i, norm_o as dicts: {label: (mean, std)}.
    Priority:
        1. /norms group (new unified format, must contain all labels)
        2. /norm_i and /norm_o groups (legacy A)
        3. root-level packed arrays (legacy B)
    """

    def _fetch_group(g, labels):
        out = {}
        for lbl in labels:
            if lbl in g:
                ds = g[lbl]
                if 'mean' in ds.attrs and 'std' in ds.attrs:
                    out[lbl] = (float(ds.attrs['mean']), float(ds.attrs['std']))
                else:
                    arr = jnp.array(ds[()])
                    out[lbl] = (float(arr[0]), float(arr[1]))
        return out

    # --- (1) Unified /norms format ---
    if _exists(nnh5, 'norms'):
        g = nnh5['norms']
        norm_i = _fetch_group(g, label_i)
        norm_o = _fetch_group(g, label_o)
        if (len(norm_i) == len(label_i)) and (len(norm_o) == len(label_o)):
            # ✅ Found full, consistent norms → use this
            return norm_i, norm_o
        else:
            # ⚠️ Warn and fall through
            print("Warning: '/norms' group incomplete; falling back to legacy groups.")

    # --- (2) Separate /norm_i and /norm_o groups ---
    if _exists(nnh5, 'norm_i') and _exists(nnh5, 'norm_o'):
        gi, go = nnh5['norm_i'], nnh5['norm_o']
        norm_i = _fetch_group(gi, label_i)
        norm_o = _fetch_group(go, label_o)
        if (len(norm_i) == len(label_i)) and (len(norm_o) == len(label_o)):
            return norm_i, norm_o
        else:
            print("Warning: '/norm_i' or '/norm_o' incomplete; falling back to packed arrays.")

    # --- (3) Root-level packed arrays ---
    if all(_exists(nnh5, k) for k in ('norm_i_mean', 'norm_i_std', 'norm_o_mean', 'norm_o_std')):
        mi, si = nnh5['norm_i_mean'][()], nnh5['norm_i_std'][()]
        mo, so = nnh5['norm_o_mean'][()], nnh5['norm_o_std'][()]
        norm_i = {lbl: (float(mi[i]), float(si[i])) for i, lbl in enumerate(label_i)}
        norm_o = {lbl: (float(mo[i]), float(so[i])) for i, lbl in enumerate(label_o)}
        return norm_i, norm_o

    raise KeyError("Could not locate valid normalization statistics in any supported layout.")


def _layernorm_last(x, gamma, beta, eps=1e-5):
    """
    Manual LayerNorm over the last dimension: (B, D) → (B, D),
    using stored gamma (scale) and beta (bias) vectors.
    """
    mu  = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mu) * (x - mu), axis=-1, keepdims=True)
    xhat = (x - mu) / jnp.sqrt(var + eps)
    return xhat * gamma + beta

    
class Net(object):
    def __init__(self, nnpath=None,nntype='MLP_v1',normed=False):
        self.normed = normed
        self.readNN(nnpath=nnpath,nntype=nntype)

    def _build_input_spec_from_teff_order(self,x_in):
            """
            Convert caller input order [Teff, logg, feh, afe, vmic, Av, Rv]
            into the file's label_i order, including Teff→log10(Teff) if
            the file expects 'logt' (or similar). Works for batch or 1D.

            This is the spectral analogue of `_build_input_from_teff_order`,
            but with vmic included.
            """
            import jax.numpy as jnp

            # ensure batch dimension
            if x_in.ndim == 1:
                x = x_in[None, :]
                squeeze = True
            else:
                x = x_in
                squeeze = False

            if x.shape[-1] != 7:
                raise ValueError(
                    f"_build_input_spec_from_teff_order expects 7 inputs "
                    f"[Teff, logg, feh, afe, vmic, Av, Rv], got shape {x.shape}"
                )

            Teff = x[:, 0]
            logg = x[:, 1]
            feh  = x[:, 2]
            afe  = x[:, 3]
            vmic = x[:, 4]
            Av   = x[:, 5]
            Rv   = x[:, 6]

            li_low = [s.lower() for s in self.label_i]
            cols = []
            for nm in li_low:
                if nm in ('teff',):
                    cols.append(Teff)
                elif nm in ('logt','logteff','log10teff'):
                    cols.append(jnp.log10(Teff))
                elif nm == 'logg':
                    cols.append(logg)
                elif nm in ('feh','[fe/h]'):
                    cols.append(feh)
                elif nm in ('afe','alpha','[a/fe]','[alpha/fe]'):
                    cols.append(afe)
                elif nm in (
                    'vmic', 'vt', 'vturb', 'vmic_turb',
                    'v_micro', 'micro', 'microturbulence', 'v_turb'
                ):
                    cols.append(vmic)
                elif nm == 'av':
                    cols.append(Av)
                elif nm == 'rv':
                    cols.append(Rv)
                else:
                    raise KeyError(
                        f"Unrecognized input label '{nm}' in file; "
                        f"cannot map from [Teff,logg,feh,afe,vmic,Av,Rv]."
                    )

            X = jnp.stack(cols, axis=-1)
            return X[0] if squeeze else X

    def readNN(self,nnpath=None,nntype='MLP_v1'):
        # read in normalization info
        nnh5 = h5py.File(nnpath,'r')

        self.label_i = [x.decode('utf-8') for x in nnh5['label_i'][()]]
        self.label_o = [x.decode('utf-8') for x in nnh5['label_o'][()]]

        self.resolution = nnh5["meta"].attrs.get('resolution', None)
        self.pixels_per_resel = nnh5["meta"].attrs.get('pixels_per_resel', None)

        # wavelengths, for convenience
        if _exists(nnh5, 'wavelengths_A'):
            self.wavelengths_A = jnp.asarray(nnh5['wavelengths_A'][()],
                                                dtype=jnp.float64)
        else:
            self.wavelengths_A = None

        # --- index map by name (lowercased contains) ---
        li_low = [s.lower() for s in self.label_i]
        def _find(*cands):
            for c in cands:
                if c in li_low:
                    return li_low.index(c)
            raise KeyError(f"Missing any of {cands} in label_i: {self.label_i}")

        self._i_logt_or_teff = _find('logt','logteff','log10teff','teff')
        self._i_logg = _find('logg')
        self._i_feh  = _find('feh','[fe/h]')
        self._i_afe  = _find('afe','alpha','[a/fe]','[alpha/fe]')
        self._i_vmic = _find(
                        'vmic', 'vt', 'vturb', 'vmic_turb',
                        'v_micro', 'micro', 'microturbulence', 'v_turb'
                    )
        self._i_av   = _find('av')
        self._i_rv   = _find('rv')
        # phys is 5-tuple in this order:
        self._i_phys = (self._i_logt_or_teff, self._i_logg, self._i_feh, self._i_afe, self._i_vmic)

        if self.normed:
            self.norm_i, self.norm_o = _read_norms(nnh5, self.label_i, self.label_o)
    
        self.D_in = len(self.label_i)
        self.D_out = len(self.label_o)

        # ===== SpectralMLP_v1 loader =====
        if (nntype == 'MLP_v1'):

            def _rp(k):
                """
                Robust parameter fetcher, same pattern as in MLP_v2:
                tries 'k', then 'k' with '.' → '/' substitution.
                """
                if k in nnh5:
                    return nnh5[k][()]
                ks = k.replace('.', '/')
                if ks in nnh5:
                    return nnh5[ks][()]
                raise KeyError(f"Param key not found: {k}")

            to32 = lambda a: jnp.asarray(a, dtype=jnp.float32)

            # ----- basis + μ_log -----
            # B: (K, L), used as FixedMatMul after toK
            Bmat = to32(_rp('model/f0.B.B'))          # (K, L)
            # μ_log: root or model-level; prefer model-level with fallback
            try:
                mu_log_model = to32(_rp('model/mu_log'))
            except KeyError:
                mu_log_model = to32(_rp('mu_log'))
            mu_log_model = mu_log_model.reshape(1, -1)  # (1, L)

            # ----- f0 (phys-only; log-flux base) -----
            b1   = to32(_rp('model/f0.lin1.bias'))
            b2   = to32(_rp('model/f0.lin2.bias'))
            b3   = to32(_rp('model/f0.lin3.bias'))
            bK   = to32(_rp('model/f0.toK.bias'))

            W1   = to32(_rp('model/f0.lin1.weight')).T   # (in, out)
            W2   = to32(_rp('model/f0.lin2.weight')).T
            W3   = to32(_rp('model/f0.lin3.weight')).T
            WK   = to32(_rp('model/f0.toK.weight')).T    # (H3, K)

            ln1b = to32(_rp('model/f0.ln1.bias'))
            ln2b = to32(_rp('model/f0.ln2.bias'))
            ln3b = to32(_rp('model/f0.ln3.bias'))
            ln1s = to32(_rp('model/f0.ln1.weight'))
            ln2s = to32(_rp('model/f0.ln2.weight'))
            ln3s = to32(_rp('model/f0.ln3.weight'))

            # ----- khat ([phys, Rv]) -----
            kb1   = to32(_rp('model/khat.lin1.bias'))
            kb2   = to32(_rp('model/khat.lin2.bias'))
            kbout = to32(_rp('model/khat.linout.bias'))
            Wk1   = to32(_rp('model/khat.lin1.weight')).T
            Wk2   = to32(_rp('model/khat.lin2.weight')).T
            Wkout = to32(_rp('model/khat.linout.weight')).T

            # ----- resid (full input) -----
            rb1   = to32(_rp('model/resid.lin1.bias'))
            rb2   = to32(_rp('model/resid.lin2.bias'))
            Wr1   = to32(_rp('model/resid.lin1.weight')).T
            Wr2   = to32(_rp('model/resid.lin2.weight')).T

            # ----- gates -----
            ext_gate = to32(_rp('model/ext_gate'))  # shape (1,)
            res_gate = to32(_rp('model/res_gate'))  # shape (1,)

            # ----- dimensions / sanity checks -----
            D_in  = self.D_in
            D_out = self.D_out                  # number of wavelength points
            d_phys = W1.shape[0]                # phys inputs for f0
            K      = Bmat.shape[0]              # latent rank
            assert W1.shape[0] == d_phys
            assert WK.shape[1] == K
            assert Bmat.shape[1] == D_out
            assert Wr1.shape[0] == D_in
            assert Wk1.shape[0] == (d_phys + 1) # [phys, Rv]

            # ----- build nnx.Linear layers -----
            # f0 (phys-only)
            f0_lin1 = nnx.Linear(W1.shape[0],  W1.shape[1],  rngs=nnx.Rngs(0)); f0_lin1.kernel = nnx.Param(W1);  f0_lin1.bias = nnx.Param(b1)
            f0_lin2 = nnx.Linear(W2.shape[0],  W2.shape[1],  rngs=nnx.Rngs(0)); f0_lin2.kernel = nnx.Param(W2);  f0_lin2.bias = nnx.Param(b2)
            f0_lin3 = nnx.Linear(W3.shape[0],  W3.shape[1],  rngs=nnx.Rngs(0)); f0_lin3.kernel = nnx.Param(W3);  f0_lin3.bias = nnx.Param(b3)
            f0_toK  = nnx.Linear(WK.shape[0],  WK.shape[1],  rngs=nnx.Rngs(0)); f0_toK.kernel  = nnx.Param(WK);  f0_toK.bias  = nnx.Param(bK)

            # khat
            kh_lin1 = nnx.Linear(Wk1.shape[0], Wk1.shape[1], rngs=nnx.Rngs(0)); kh_lin1.kernel = nnx.Param(Wk1); kh_lin1.bias = nnx.Param(kb1)
            kh_lin2 = nnx.Linear(Wk2.shape[0], Wk2.shape[1], rngs=nnx.Rngs(0)); kh_lin2.kernel = nnx.Param(Wk2); kh_lin2.bias = nnx.Param(kb2)
            kh_out  = nnx.Linear(Wkout.shape[0],Wkout.shape[1],rngs=nnx.Rngs(0)); kh_out.kernel = nnx.Param(Wkout);kh_out.bias = nnx.Param(kbout)

            # resid
            rs_lin1 = nnx.Linear(Wr1.shape[0], Wr1.shape[1], rngs=nnx.Rngs(0)); rs_lin1.kernel = nnx.Param(Wr1); rs_lin1.bias = nnx.Param(rb1)
            rs_out  = nnx.Linear(Wr2.shape[0], Wr2.shape[1], rngs=nnx.Rngs(0)); rs_out.kernel  = nnx.Param(Wr2); rs_out.bias  = nnx.Param(rb2)

            # store everything we need for forward
            self._spec_ln_params = ((ln1s, ln1b), (ln2s, ln2b), (ln3s, ln3b))
            self._spec_f0_layers = (f0_lin1, f0_lin2, f0_lin3, f0_toK, Bmat)
            self._spec_kh_layers = (kh_lin1, kh_lin2, kh_out)
            self._spec_rs_layers = (rs_lin1, rs_out)

            self._spec_mu_log        = mu_log_model                   # (1, D_out)
            self._spec_ext_gate_raw  = ext_gate                       # (1,)
            self._spec_res_gate_raw  = res_gate                       # (1,)
            self._spec_max_khat      = 10.0   # same defaults as PyTorch
            self._spec_max_resid_dex = 0.5


            # ----- core forward in *normalized input* space -----
            def _forward_spec(x_file_norm):
                """
                x_file_norm: input in file's label_i order, normalized by norm_i
                             (i.e., exactly what the PyTorch model saw).

                Returns: log10 flux array (B, D_out), already including μ_log
                         and extinction+residual lanes.
                """
                import jax
                import jax.numpy as jnp

                x_file_norm = jnp.asarray(x_file_norm, dtype=jnp.float32)
                single = (x_file_norm.ndim == 1)
                if single:
                    x_file_norm = x_file_norm[None, :]

                Bsz = x_file_norm.shape[0]

                # indices: phys slice, Av, Rv (already set earlier in readNN)
                phys = x_file_norm[:, jnp.array(self._i_phys)]  # (B, d_phys)

                # Av, Rv as *normalized* inputs (matches training)
                Av = x_file_norm[:, self._i_av]   if hasattr(self, '_i_av') else jnp.zeros((Bsz,), dtype=x_file_norm.dtype)
                Rv = x_file_norm[:, self._i_rv]   if hasattr(self, '_i_rv') else jnp.full((Bsz,), 3.1, dtype=x_file_norm.dtype)

                # ----- stellar head f0: (Linear → SiLU → LN) ×3 → toK → B -----
                (f0_lin1, f0_lin2, f0_lin3, f0_toK, Bmat_) = self._spec_f0_layers
                (g1, b1_), (g2, b2_), (g3, b3_) = self._spec_ln_params

                z = nnx.silu(f0_lin1(phys))
                z = _layernorm_last(z, g1, b1_)
                z = nnx.silu(f0_lin2(z))
                z = _layernorm_last(z, g2, b2_)
                z = nnx.silu(f0_lin3(z))
                z = _layernorm_last(z, g3, b3_)
                zK = f0_toK(z)                # (B, K)
                base_log = zK @ Bmat_         # (B, D_out)
                base_log = base_log + self._spec_mu_log  # add μ_log

                # ----- extinction lane: khat([phys,Rv]) -----
                (kh_lin1, kh_lin2, kh_out) = self._spec_kh_layers
                kh_in = jnp.concatenate([phys, Rv[:, None]], axis=-1)
                zk = nnx.silu(kh_lin1(kh_in))
                zk = nnx.silu(kh_lin2(zk))
                k_hat = jax.nn.softplus(kh_out(zk))  # (B, D_out) ≥ 0

                max_k = self._spec_max_khat
                if max_k is not None:
                    k_hat = jnp.minimum(k_hat, max_k)

                # ----- residual lane: resid(full x) -----
                (rs_lin1, rs_out) = self._spec_rs_layers
                r_hat = rs_out(nnx.silu(rs_lin1(x_file_norm)))  # (B, D_out)

                max_r = self._spec_max_resid_dex
                if max_r is not None:
                    r_hat = jnp.clip(r_hat, -max_r, max_r)

                # ----- gates -----
                g_ext = jax.nn.sigmoid(self._spec_ext_gate_raw)  # (1,)
                g_res = jax.nn.sigmoid(self._spec_res_gate_raw)  # (1,)

                # Av here is still normalized (matches how PyTorch saw it)
                ext_term = (-0.4 * Av[:, None]) * k_hat * g_ext
                r_hat = r_hat * g_res

                y_log = base_log + ext_term + r_hat
                return y_log[0] if single else y_log

            self._eval_core_spec = _forward_spec
            self.eval = self.evalSpec

    def evalSpec(self, x):
        """
        Evaluate SpectralMLP_v1 model.

        Parameters
        ----------
        x : array-like
            Canonical spectral input order:
                [Teff, logg, FeH, aFe, vmic, Av, Rv]
            or (B, 7) batch of those.

        Returns
        -------
        y_den : jnp.ndarray
            Predicted log10(flux) per wavelength in label_o order.
            If `self.normed` is True and output norms exist, this is
            *denormalized* using norm_o (i.e., back to the training
            log-flux scale).
        """

        x_i = jnp.asarray(x, dtype=jnp.float32)
        single = (x_i.ndim == 1)

        # 1) caller → file input order (Teff→log10 if needed, vmic mapped)
        x_for_file = self._build_input_spec_from_teff_order(x_i)

        # 2) normalize inputs with dict norms (label_i order)
        if self.normed:
            mi = jnp.array([self.norm_i[l][0] for l in self.label_i], dtype=jnp.float32)
            si = jnp.array([self.norm_i[l][1] for l in self.label_i], dtype=jnp.float32)
            si = jnp.where(si == 0.0, 1.0, si)
            if single:
                x_norm = (x_for_file - mi) / si
            else:
                x_norm = (x_for_file - mi[None, :]) / si[None, :]
        else:
            x_norm = x_for_file

        # 3) forward core (already in model order & normalized)
        y = self._eval_core_spec(x_norm)   # (B,D_out) or (D_out,) if single

        # 4) denormalize outputs with dict norms (label_o order)
        if self.normed:
            mo = jnp.array([self.norm_o[l][0] for l in self.label_o], dtype=jnp.float32)
            so = jnp.array([self.norm_o[l][1] for l in self.label_o], dtype=jnp.float32)
            if y.ndim == 1:
                y_den = y * so + mo
            else:
                y_den = y * so[None, :] + mo[None, :]
        else:
            y_den = y

        # 5) squeeze if single
        if single and (y_den.ndim == 2) and (y_den.shape[0] == 1):
            return y_den[0]
        return y_den

class modpred(object):
    """docstring for modpred"""
    def __init__(self, nnpath=None, nntype='MLP_v0', norm=True):
        super(modpred, self).__init__()
        
        if nnpath != None:
            self.nnpath = nnpath
        else:
            raise IOError('... Must provide a path to the ANN model')

        self.normed = norm

        self.anns = Net(nnpath=self.nnpath,nntype=nntype,normed=self.normed)
        self.inlabels   = self.anns.label_i
        self.outlabels  = self.anns.label_o
        self.wavelength = self.anns.wavelengths_A
        self.resolution = self.anns.resolution

    def pred(self,inpars):
        # model is in F_nu, convert to F_lambda
        C_A_PER_S = 2.99792458e18
        f_nu = 10.0**self.anns.evalSpec(inpars)
        f_lambda = f_nu * (C_A_PER_S / (self.wavelength ** 2)) * (4.0 * jnp.pi)
        return f_lambda
    
    def predspec(self, pars):
        """
        pars: [Teff, logg, feh, afe, vmic, Av, Rv]
        """

        if isinstance(pars,list):
            pars = jnp.asarray(pars)
        
        # make copy of input array so that the code doesn't change inplace
        pars = jnp.copy(pars)

        # make the prediction
        y_i = self.pred(pars)        
        
        return y_i