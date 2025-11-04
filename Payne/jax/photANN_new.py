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


class Net(object):
    def __init__(self, nnpath=None,nntype='MLP_v1',normed=True):
        self.normed = normed
        self.readNN(nnpath=nnpath,nntype=nntype)

    def _build_input_from_teff_order(self, x_in):
        """
        Convert caller input order [Teff, logg, feh, afe, Av, Rv]
        into the file's label_i order, including Teff→log10(Teff) if
        the file expects 'logt' (or similar). Works for batch or 1D.
        """
        import jax.numpy as jnp

        # ensure batch dimension
        if x_in.ndim == 1:
            x = x_in[None, :]
            squeeze = True
        else:
            x = x_in
            squeeze = False

        Teff = x[:, 0]
        logg = x[:, 1]
        feh  = x[:, 2]
        afe  = x[:, 3]
        Av   = x[:, 4]
        Rv   = x[:, 5]

        # build by file-declared input labels
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
            elif nm == 'av':
                cols.append(Av)
            elif nm == 'rv':
                cols.append(Rv)
            else:
                raise KeyError(f"Unrecognized input label '{nm}' in file; "
                            f"cannot map from [Teff,logg,feh,afe,Av,Rv].")
        X = jnp.stack(cols, axis=-1)
        return X[0] if squeeze else X

    def readNN(self,nnpath=None,nntype='MLP_v1'):
        # read in normalization info
        nnh5 = h5py.File(nnpath,'r')

        self.label_i = [x.decode('utf-8') for x in nnh5['label_i'][()]]
        self.label_o = [x.decode('utf-8') for x in nnh5['label_o'][()]]

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
        self._i_av   = _find('av')
        self._i_rv   = _find('rv')
        # phys is 4-tuple in this order:
        self._i_phys = (self._i_logt_or_teff, self._i_logg, self._i_feh, self._i_afe)

        if self.normed:
            self.norm_i, self.norm_o = _read_norms(nnh5, self.label_i, self.label_o)
    
        self.D_in = len(self.label_i)
        self.D_out = len(self.label_o)

        if (nntype == 'MLP_v0'):
            self.bias1 = jnp.array(nnh5['model/mlp.lin1.bias'][()])
            self.bias2 = jnp.array(nnh5['model/mlp.lin2.bias'][()])
            self.bias3 = jnp.array(nnh5['model/mlp.lin3.bias'][()])
            self.bias4 = jnp.array(nnh5['model/mlp.lin4.bias'][()])
            self.bias5 = jnp.array(nnh5['model/mlp.lin5.bias'][()])
            self.bias6 = jnp.array(nnh5['model/mlp.lin6.bias'][()])
            self.bias7 = jnp.array(nnh5['model/mlp.lin7.bias'][()])
            self.bias8 = jnp.array(nnh5['model/mlp.lin8.bias'][()])
            self.bias9 = jnp.array(nnh5['model/mlp.lin9.bias'][()])

            self.weight1 = jnp.transpose(jnp.array(nnh5['model/mlp.lin1.weight'][()]),(1,0))
            self.weight2 = jnp.transpose(jnp.array(nnh5['model/mlp.lin2.weight'][()]),(1,0))
            self.weight3 = jnp.transpose(jnp.array(nnh5['model/mlp.lin3.weight'][()]),(1,0))
            self.weight4 = jnp.transpose(jnp.array(nnh5['model/mlp.lin4.weight'][()]),(1,0))
            self.weight5 = jnp.transpose(jnp.array(nnh5['model/mlp.lin5.weight'][()]),(1,0))
            self.weight6 = jnp.transpose(jnp.array(nnh5['model/mlp.lin6.weight'][()]),(1,0))
            self.weight7 = jnp.transpose(jnp.array(nnh5['model/mlp.lin7.weight'][()]),(1,0))
            self.weight8 = jnp.transpose(jnp.array(nnh5['model/mlp.lin8.weight'][()]),(1,0))
            self.weight9 = jnp.transpose(jnp.array(nnh5['model/mlp.lin9.weight'][()]),(1,0))

            self.lin1 = nnx.Linear(in_features=self.weight1.shape[0],out_features=self.weight1.shape[1],rngs=nnx.Rngs(0))
            self.lin1.kernel = nnx.Param(value=self.weight1)
            self.lin1.bias = nnx.Param(value=self.bias1)

            self.lin2 = nnx.Linear(in_features=self.weight2.shape[0],out_features=self.weight2.shape[1],rngs=nnx.Rngs(0))
            self.lin2.kernel = nnx.Param(value=self.weight2)
            self.lin2.bias = nnx.Param(value=self.bias2)

            self.lin3 = nnx.Linear(in_features=self.weight3.shape[0],out_features=self.weight3.shape[1],rngs=nnx.Rngs(0))
            self.lin3.kernel = nnx.Param(value=self.weight3)
            self.lin3.bias = nnx.Param(value=self.bias3)

            self.lin4 = nnx.Linear(in_features=self.weight4.shape[0],out_features=self.weight4.shape[1],rngs=nnx.Rngs(0))
            self.lin4.kernel = nnx.Param(value=self.weight4)
            self.lin4.bias = nnx.Param(value=self.bias4)

            self.lin5 = nnx.Linear(in_features=self.weight5.shape[0],out_features=self.weight5.shape[1],rngs=nnx.Rngs(0))
            self.lin5.kernel = nnx.Param(value=self.weight5)
            self.lin5.bias = nnx.Param(value=self.bias5)

            self.lin6 = nnx.Linear(in_features=self.weight6.shape[0],out_features=self.weight6.shape[1],rngs=nnx.Rngs(0))
            self.lin6.kernel = nnx.Param(value=self.weight6)
            self.lin6.bias = nnx.Param(value=self.bias6)

            self.lin7 = nnx.Linear(in_features=self.weight7.shape[0],out_features=self.weight7.shape[1],rngs=nnx.Rngs(0))
            self.lin7.kernel = nnx.Param(value=self.weight7)
            self.lin7.bias = nnx.Param(value=self.bias7)

            self.lin8 = nnx.Linear(in_features=self.weight8.shape[0],out_features=self.weight8.shape[1],rngs=nnx.Rngs(0))
            self.lin8.kernel = nnx.Param(value=self.weight8)
            self.lin8.bias = nnx.Param(value=self.bias8)

            self.lin9 = nnx.Linear(in_features=self.weight9.shape[0],out_features=self.weight9.shape[1],rngs=nnx.Rngs(0))
            self.lin9.kernel = nnx.Param(value=self.weight9)
            self.lin9.bias = nnx.Param(value=self.bias9)
            
            self.mlp = nnx.Sequential(
                self.lin1,
                nnx.gelu,
                self.lin2,
                nnx.gelu,
                self.lin3,
                nnx.gelu,
                self.lin4,
                nnx.gelu,
                self.lin5,
                nnx.gelu,
                self.lin6,
                nnx.gelu,
                self.lin7,
                nnx.gelu,
                self.lin8,
                nnx.gelu,
                self.lin9,
            )

        if (nntype == 'MLP_v1'):
            
            # read in the weights and biases from the HDF5 file
            bias1 = jnp.array(nnh5['model/mlp.lin1.bias'][()])
            bias2 = jnp.array(nnh5['model/mlp.lin2.bias'][()])
            bias3 = jnp.array(nnh5['model/mlp.lin3.bias'][()])
            biasout = jnp.array(nnh5['model/mlp.linout.bias'][()])

            weight1 = jnp.transpose(jnp.array(nnh5['model/mlp.lin1.weight'][()]),(1,0))
            weight2 = jnp.transpose(jnp.array(nnh5['model/mlp.lin2.weight'][()]),(1,0))
            weight3 = jnp.transpose(jnp.array(nnh5['model/mlp.lin3.weight'][()]),(1,0))
            weightout = jnp.transpose(jnp.array(nnh5['model/mlp.linout.weight'][()]),(1,0))

            ln1bias = jnp.array(nnh5['model/mlp.ln1.bias'][()])
            ln2bias = jnp.array(nnh5['model/mlp.ln2.bias'][()])
            ln3bias = jnp.array(nnh5['model/mlp.ln3.bias'][()])
            ln1scale = jnp.array(nnh5['model/mlp.ln1.weight'][()])
            ln2scale = jnp.array(nnh5['model/mlp.ln2.weight'][()])
            ln3scale = jnp.array(nnh5['model/mlp.ln3.weight'][()])

            # create the layers, setting the weights and biases to the values read in from the HDF5 file
            lin1 = nnx.Linear(in_features=weight1.shape[0],out_features=weight1.shape[1],rngs=nnx.Rngs(0))
            lin1.kernel = nnx.Param(value=weight1)
            lin1.bias = nnx.Param(value=bias1)

            lin2 = nnx.Linear(in_features=weight2.shape[0],out_features=weight2.shape[1],rngs=nnx.Rngs(0))
            lin2.kernel = nnx.Param(value=weight2)
            lin2.bias = nnx.Param(value=bias2)

            lin3 = nnx.Linear(in_features=weight3.shape[0],out_features=weight3.shape[1],rngs=nnx.Rngs(0))
            lin3.kernel = nnx.Param(value=weight3)
            lin3.bias = nnx.Param(value=bias3)

            linout = nnx.Linear(in_features=weightout.shape[0],out_features=weightout.shape[1],rngs=nnx.Rngs(0))
            linout.kernel = nnx.Param(value=weightout)
            linout.bias = nnx.Param(value=biasout)

            ln1 = nnx.LayerNorm(num_features=lin1.bias.shape[0],rngs=nnx.Rngs(0))
            ln1.bias = nnx.Param(value=ln1bias)
            ln1.scale = nnx.Param(value=ln1scale)
            
            ln2 = nnx.LayerNorm(num_features=lin2.bias.shape[0],rngs=nnx.Rngs(0))
            ln2.bias = nnx.Param(value=ln2bias)
            ln2.scale = nnx.Param(value=ln2scale)
                        
            ln3 = nnx.LayerNorm(num_features=lin3.bias.shape[0],rngs=nnx.Rngs(0))
            ln3.bias = nnx.Param(value=ln3bias)
            ln3.scale = nnx.Param(value=ln3scale)

            # define the network
            self.mlp = nnx.Sequential(
                lin1,
                ln1,
                nnx.silu,
                lin2,
                ln2,
                nnx.silu,
                lin3,
                ln3,
                nnx.silu,
                linout
            )

            self.eval = self.evalMLP

        # ===== MLP_v2 loader =====
        if (nntype == 'MLP_v2'):

            def _rp(k):
                if k in nnh5: return nnh5[k][()]
                ks = k.replace('.', '/')
                if ks in nnh5: return nnh5[ks][()]
                raise KeyError(f"Param key not found: {k}")

            to32 = lambda a: jnp.asarray(a, dtype=jnp.float32)

            # ----- f0 (phys-only) -----
            b1    = to32(_rp('model/f0.lin1.bias'))
            b2    = to32(_rp('model/f0.lin2.bias'))
            b3    = to32(_rp('model/f0.lin3.bias'))
            bout  = to32(_rp('model/f0.linout.bias'))
            W1    = to32(_rp('model/f0.lin1.weight')).T      # (out,in)->(in,out)
            W2    = to32(_rp('model/f0.lin2.weight')).T
            W3    = to32(_rp('model/f0.lin3.weight')).T
            Wout  = to32(_rp('model/f0.linout.weight')).T
            ln1b  = to32(_rp('model/f0.ln1.bias'))
            ln2b  = to32(_rp('model/f0.ln2.bias'))
            ln3b  = to32(_rp('model/f0.ln3.bias'))
            ln1s  = to32(_rp('model/f0.ln1.weight'))
            ln2s  = to32(_rp('model/f0.ln2.weight'))
            ln3s  = to32(_rp('model/f0.ln3.weight'))

            # ----- khat ([phys,Rv]) -----
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

            # Dimensions (safety)
            d_phys = 4
            D_in   = self.D_in
            D_out  = self.D_out
            assert W1.shape[0] == d_phys
            assert Wout.shape[1] == D_out
            assert Wr1.shape[0] == D_in
            assert Wk1.shape[0] == (d_phys + 1)

            # ----- build layers (nnx.Linear expects (in,out)) -----
            f0_lin1 = nnx.Linear(W1.shape[0],   W1.shape[1],   rngs=nnx.Rngs(0)); f0_lin1.kernel = nnx.Param(W1);   f0_lin1.bias = nnx.Param(b1)
            f0_lin2 = nnx.Linear(W2.shape[0],   W2.shape[1],   rngs=nnx.Rngs(0)); f0_lin2.kernel = nnx.Param(W2);   f0_lin2.bias = nnx.Param(b2)
            f0_lin3 = nnx.Linear(W3.shape[0],   W3.shape[1],   rngs=nnx.Rngs(0)); f0_lin3.kernel = nnx.Param(W3);   f0_lin3.bias = nnx.Param(b3)
            f0_out  = nnx.Linear(Wout.shape[0], Wout.shape[1], rngs=nnx.Rngs(0)); f0_out.kernel  = nnx.Param(Wout); f0_out.bias  = nnx.Param(bout)

            kh_lin1 = nnx.Linear(Wk1.shape[0],  Wk1.shape[1],  rngs=nnx.Rngs(0)); kh_lin1.kernel = nnx.Param(Wk1);  kh_lin1.bias = nnx.Param(kb1)
            kh_lin2 = nnx.Linear(Wk2.shape[0],  Wk2.shape[1],  rngs=nnx.Rngs(0)); kh_lin2.kernel = nnx.Param(Wk2);  kh_lin2.bias = nnx.Param(kb2)
            kh_out  = nnx.Linear(Wkout.shape[0],Wkout.shape[1],rngs=nnx.Rngs(0)); kh_out.kernel  = nnx.Param(Wkout);kh_out.bias  = nnx.Param(kbout)

            rs_lin1 = nnx.Linear(Wr1.shape[0],  Wr1.shape[1],  rngs=nnx.Rngs(0)); rs_lin1.kernel = nnx.Param(Wr1);  rs_lin1.bias = nnx.Param(rb1)
            rs_out  = nnx.Linear(Wr2.shape[0],  Wr2.shape[1],  rngs=nnx.Rngs(0)); rs_out.kernel  = nnx.Param(Wr2);  rs_out.bias  = nnx.Param(rb2)

            # store LN params (functional)
            self._ln_params = ((ln1s, ln1b), (ln2s, ln2b), (ln3s, ln3b))
            self._f0_layers = (f0_lin1, f0_lin2, f0_lin3, f0_out)
            self._kh_layers = (kh_lin1, kh_lin2, kh_out)
            self._rs_layers = (rs_lin1, rs_out)

            # ----- forward core (predict-time: no dropout) -----
            def _layernorm_last(x, gamma, beta, eps=1e-5):
                mu  = jnp.mean(x, axis=-1, keepdims=True)
                var = jnp.mean((x - mu) * (x - mu), axis=-1, keepdims=True)  # unbiased=False
                xhat = (x - mu) / jnp.sqrt(var + eps)
                return xhat * gamma + beta

            def _forward_v2(x_file_norm):
                if x_file_norm.ndim == 1:
                    x_file_norm = x_file_norm[None, :]

                # slices (already in file input order)
                phys = x_file_norm[:, jnp.array(self._i_phys)]            # (B,4)
                Av   = x_file_norm[:, self._i_av:self._i_av+1]            # (B,1)
                Rv   = x_file_norm[:, self._i_rv:self._i_rv+1]            # (B,1)

                # f0: (Linear → SiLU → LN) × 3 → Linear
                (f0_lin1, f0_lin2, f0_lin3, f0_out) = self._f0_layers
                (g1,b1_), (g2,b2_), (g3,b3_) = self._ln_params
                z = nnx.silu(f0_lin1(phys))
                z = _layernorm_last(z, g1, b1_)
                z = nnx.silu(f0_lin2(z))
                z = _layernorm_last(z, g2, b2_)
                z = nnx.silu(f0_lin3(z))
                z = _layernorm_last(z, g3, b3_)
                bc0 = f0_out(z)  # (B, D_out)

                # khat([phys, Rv]) → softplus
                (kh_lin1, kh_lin2, kh_out) = self._kh_layers
                zk = nnx.silu(kh_lin1(jnp.concatenate([phys, Rv], axis=-1)))
                zk = nnx.silu(kh_lin2(zk))
                k_hat = jax.nn.softplus(kh_out(zk))                       # (B, D_out)

                # resid(full x): Linear → SiLU → Linear
                (rs_lin1, rs_out) = self._rs_layers
                r_hat = rs_out(nnx.silu(rs_lin1(x_file_norm)))            # (B, D_out)

                # compose (M_bol − M_band): extinction lowers BC
                return bc0 + r_hat - Av * k_hat

            self._eval_core_v2 = _forward_v2
            self.eval = self.evalMLP_v2    
        nnh5.close()
        
        
    def evalMLP(self, x):
        x_i = jnp.asarray(x, dtype=jnp.float32)
        single = (x_i.ndim == 1)

        if self.normed:
            mi = jnp.array([self.norm_i[l][0] for l in self.label_i], dtype=jnp.float32)
            si = jnp.array([self.norm_i[l][1] for l in self.label_i], dtype=jnp.float32)
            si = jnp.where(si == 0.0, 1.0, si)
            x_norm = (x_i - mi) / si if single else (x_i - mi[None, :]) / si[None, :]
        else:
            x_norm = x_i

        y = self.mlp(x_norm)

        if self.normed:
            mo = jnp.array([self.norm_o[l][0] for l in self.label_o], dtype=jnp.float32)
            so = jnp.array([self.norm_o[l][1] for l in self.label_o], dtype=jnp.float32)
            y = y * so + mo if y.ndim == 1 else (y * so[None, :] + mo[None, :])

        return y
 

    def evalMLP_v2(self, x):
        """
        x: [Teff, logg, FeH, aFe, Av, Rv] or (B,6)
        Returns denormalized BCs in label_o order, matching PyTorch MLP_v2.
        """
        x_i = jnp.asarray(x, dtype=jnp.float32)
        single = (x_i.ndim == 1)

        # 1) caller → file input order (Teff→log10 if needed)
        x_for_file = self._build_input_from_teff_order(x_i)  # (6,) or (B,6) in label_i order

        # 2) normalize inputs with dict norms (label_i order)
        if self.normed:
            mi = jnp.array([self.norm_i[l][0] for l in self.label_i], dtype=jnp.float32)
            si = jnp.array([self.norm_i[l][1] for l in self.label_i], dtype=jnp.float32)
            si = jnp.where(si == 0.0, 1.0, si)
            x_norm = (x_for_file - mi) / si if single else (x_for_file - mi[None, :]) / si[None, :]
        else:
            x_norm = x_for_file
 
        # 3) forward core (already in model order)
        y = self._eval_core_v2(x_norm)

        # 4) denormalize outputs with dict norms (label_o order)
        if self.normed:
            mo = jnp.array([self.norm_o[l][0] for l in self.label_o], dtype=jnp.float32)
            so = jnp.array([self.norm_o[l][1] for l in self.label_o], dtype=jnp.float32)
            y_den = y * so + mo if y.ndim == 1 else y * so[None, :] + mo[None, :]
        else:
            y_den = y

        # squeeze if single
        return y_den if not (single and y_den.ndim == 2 and y_den.shape[0] == 1) else y_den[0]
    
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

        self.modpararr = self.anns.label_o
        
    def pred(self,inpars):
        return self.anns.eval(inpars)
    
    def getbc(self,pars):
        if isinstance(pars,list):
            pars = jnp.asarray(pars)
        
        # make copy of input array so that the code doesn't change inplace
        pars = jnp.copy(pars)
        
        modpred = self.pred(pars)
        
        out = {}
        
        # make output dictionary        
        if len(pars.shape) == 1:
            out_i = {y:modpred[ii] for ii,y in enumerate(self.anns.label_o)}
            out.update(out_i)
        else:
            out_i = {y:modpred[:,ii] for ii,y in enumerate(self.anns.label_o)}
            out.update(out_i)
        
        return out
