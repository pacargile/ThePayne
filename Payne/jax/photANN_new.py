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
    Return (norm_i, norm_o) lists where each element is jnp.array([mean, std]).
    Supports:
      • NEW (your file):   norms/<label> with attrs {'mean','std'}
      • Legacy A:          norm_i/<label>, norm_o/<label> datasets (value = [mean,std])
      • Legacy B:          packed arrays: norm_i_mean, norm_i_std, norm_o_mean, norm_o_std
      • Legacy C:          subgroup packed: norm_i/{mean,std}, norm_o/{mean,std}
    """
    import jax.numpy as jnp

    # --- try NEW layout first ---
    if _exists(nnh5, 'norms'):
        g = nnh5['norms']
        def _fetch(label):
            if label in g:
                ds = g[label]
                # attributes required: mean, std
                if 'mean' in ds.attrs and 'std' in ds.attrs:
                    return jnp.array([ds.attrs['mean'], ds.attrs['std']])
            raise KeyError(f"norms/{label} missing or lacks attrs ['mean','std']")
        try:
            norm_i = [ _fetch(lbl) for lbl in label_i ]
            norm_o = [ _fetch(lbl) for lbl in label_o ]
            return norm_i, norm_o
        except KeyError:
            pass  # fall through to legacy paths

    # --- Legacy A: per-label datasets under norm_i/ and norm_o/ ---
    if _exists(nnh5, 'norm_i') and _exists(nnh5, 'norm_o'):
        gi, go = nnh5['norm_i'], nnh5['norm_o']
        try:
            norm_i = [ jnp.array(gi[lbl][()]) for lbl in label_i ]
            norm_o = [ jnp.array(go[lbl][()]) for lbl in label_o ]
            return norm_i, norm_o
        except Exception:
            # maybe subgroup packed:
            if all(_exists(gi, k) for k in ('mean', 'std')) and all(_exists(go, k) for k in ('mean', 'std')):
                mi, si = gi['mean'][()], gi['std'][()]
                mo, so = go['mean'][()], go['std'][()]
                norm_i = [ jnp.array([mi[i], si[i]]) for i in range(len(label_i)) ]
                norm_o = [ jnp.array([mo[i], so[i]]) for i in range(len(label_o)) ]
                return norm_i, norm_o
            # else keep falling through

    # --- Legacy B: root-level packed arrays ---
    if all(_exists(nnh5, k) for k in ('norm_i_mean','norm_i_std','norm_o_mean','norm_o_std')):
        mi, si = nnh5['norm_i_mean'][()], nnh5['norm_i_std'][()]
        mo, so = nnh5['norm_o_mean'][()], nnh5['norm_o_std'][()]
        norm_i = [ jnp.array([mi[i], si[i]]) for i in range(len(label_i)) ]
        norm_o = [ jnp.array([mo[i], so[i]]) for i in range(len(label_o)) ]
        return norm_i, norm_o

    raise KeyError("Could not locate normalization statistics in any supported layout.")


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
            import jax.numpy as jnp

            def _rp(k):  # read param helper (works with dotted or slashed)
                if k in nnh5: return nnh5[k][()]
                ks = k.replace('.', '/')
                if ks in nnh5: return nnh5[ks][()]
                raise KeyError(f"Param key not found: {k}")

            def _as_in_out(W, expected_in):
                W = jnp.asarray(W)
                if W.ndim != 2:
                    raise ValueError(f"Linear weight must be 2D, got {W.shape}")
                if W.shape[0] == expected_in:
                    return W
                if W.shape[1] == expected_in:
                    return W.T
                raise ValueError(f"Weight shape {W.shape} incompatible with expected_in={expected_in}")



            # ----- f0 -----
            b1   = jnp.array(_rp('model/f0.lin1.bias'));   W1   = jnp.array(_rp('model/f0.lin1.weight'))
            b2   = jnp.array(_rp('model/f0.lin2.bias'));   W2   = jnp.array(_rp('model/f0.lin2.weight'))
            b3   = jnp.array(_rp('model/f0.lin3.bias'));   W3   = jnp.array(_rp('model/f0.lin3.weight'))
            bout = jnp.array(_rp('model/f0.linout.bias')); Wout = jnp.array(_rp('model/f0.linout.weight'))
            ln1b = jnp.array(_rp('model/f0.ln1.bias'));    ln1s = jnp.array(_rp('model/f0.ln1.weight'))
            ln2b = jnp.array(_rp('model/f0.ln2.bias'));    ln2s = jnp.array(_rp('model/f0.ln2.weight'))
            ln3b = jnp.array(_rp('model/f0.ln3.bias'));    ln3s = jnp.array(_rp('model/f0.ln3.weight'))

            # ----- khat (new two-hidden + softplus) -----
            kb1   = jnp.array(_rp('model/khat.lin1.bias'));   kW1   = jnp.array(_rp('model/khat.lin1.weight'))
            kb2   = jnp.array(_rp('model/khat.lin2.bias'));   kW2   = jnp.array(_rp('model/khat.lin2.weight'))
            kbout = jnp.array(_rp('model/khat.linout.bias')); kWout = jnp.array(_rp('model/khat.linout.weight'))

            # ----- resid -----
            rb1  = jnp.array(_rp('model/resid.lin1.bias'));  rW1 = jnp.array(_rp('model/resid.lin1.weight'))
            rb2  = jnp.array(_rp('model/resid.lin2.bias'));  rW2 = jnp.array(_rp('model/resid.lin2.weight'))

            # Compute Expected Input Sizes and Orientations
            d_phys = 4                               # model definition
            H1, H2, H3 = b1.shape[0], b2.shape[0], b3.shape[0]

            W1   = _as_in_out(W1,   d_phys)
            W2   = _as_in_out(W2,   H1)
            W3   = _as_in_out(W3,   H2)
            Wout = _as_in_out(Wout, H3)

            # khat expected ins
            d_khat_in = d_phys + 1                   # phys + Rv
            Wk1   = _as_in_out(kW1,   d_khat_in)
            Wk2   = _as_in_out(kW2,   kb1.shape[0])  # next layer in = prev out = len(bias)
            Wkout = _as_in_out(kWout, kb2.shape[0])

            # resid expected ins
            d_full = self.D_in                       # full input vector length from file
            Wr1 = _as_in_out(rW1, d_full)
            Wr2 = _as_in_out(rW2, rb1.shape[0])

            # ----- build layers (nnx) -----
            # f0
            f0_lin1 = nnx.Linear(W1.shape[0], W1.shape[1], rngs=nnx.Rngs(0)); f0_lin1.kernel = nnx.Param(W1);   f0_lin1.bias = nnx.Param(b1)
            f0_ln1  = nnx.LayerNorm(f0_lin1.bias.shape[0], rngs=nnx.Rngs(0));  f0_ln1.bias    = nnx.Param(ln1b); f0_ln1.scale = nnx.Param(ln1s)
            f0_lin2 = nnx.Linear(W2.shape[0], W2.shape[1], rngs=nnx.Rngs(0)); f0_lin2.kernel = nnx.Param(W2);   f0_lin2.bias = nnx.Param(b2)
            f0_ln2  = nnx.LayerNorm(f0_lin2.bias.shape[0], rngs=nnx.Rngs(0));  f0_ln2.bias    = nnx.Param(ln2b); f0_ln2.scale = nnx.Param(ln2s)
            f0_lin3 = nnx.Linear(W3.shape[0], W3.shape[1], rngs=nnx.Rngs(0)); f0_lin3.kernel = nnx.Param(W3);   f0_lin3.bias = nnx.Param(b3)
            f0_ln3  = nnx.LayerNorm(f0_lin3.bias.shape[0], rngs=nnx.Rngs(0));  f0_ln3.bias    = nnx.Param(ln3b); f0_ln3.scale = nnx.Param(ln3s)
            f0_out  = nnx.Linear(Wout.shape[0], Wout.shape[1], rngs=nnx.Rngs(0)); f0_out.kernel = nnx.Param(Wout); f0_out.bias = nnx.Param(bout)

            # khat (no LN; 2 hidden; softplus in forward)
            kh_lin1 = nnx.Linear(Wk1.shape[0], Wk1.shape[1], rngs=nnx.Rngs(0)); kh_lin1.kernel = nnx.Param(Wk1); kh_lin1.bias = nnx.Param(kb1)
            kh_lin2 = nnx.Linear(Wk2.shape[0], Wk2.shape[1], rngs=nnx.Rngs(0)); kh_lin2.kernel = nnx.Param(Wk2); kh_lin2.bias = nnx.Param(kb2)
            kh_out  = nnx.Linear(Wkout.shape[0], Wkout.shape[1], rngs=nnx.Rngs(0)); kh_out.kernel = nnx.Param(Wkout); kh_out.bias = nnx.Param(kbout)

            # resid
            rs_lin1 = nnx.Linear(Wr1.shape[0], Wr1.shape[1], rngs=nnx.Rngs(0)); rs_lin1.kernel = nnx.Param(Wr1); rs_lin1.bias = nnx.Param(rb1)
            rs_out  = nnx.Linear(Wr2.shape[0], Wr2.shape[1], rngs=nnx.Rngs(0)); rs_out.kernel  = nnx.Param(Wr2); rs_out.bias  = nnx.Param(rb2)

            self._f0_layers = (f0_lin1, f0_ln1, f0_lin2, f0_ln2, f0_lin3, f0_ln3, f0_out)
            self._kh_layers = (kh_lin1, kh_lin2, kh_out)
            self._rs_layers = (rs_lin1, rs_out)

            # ----- forward core (predict-time: no dropout) -----
            def _forward_v2(x_file_norm):
                import jax.numpy as jnp
                if x_file_norm.ndim == 1:
                    x_file_norm = x_file_norm[None, :]

                # select by indices we computed above
                phys = x_file_norm[:, jnp.array(self._i_phys)]
                Av   = x_file_norm[:, self._i_av:self._i_av+1]
                Rv   = x_file_norm[:, self._i_rv:self._i_rv+1]

                # f0(phys): Linear → SiLU → LayerNorm  (×3), then linout
                f0_lin1, f0_ln1, f0_lin2, f0_ln2, f0_lin3, f0_ln3, f0_out = self._f0_layers
                z = f0_lin1(phys); z = nnx.silu(z); z = f0_ln1(z)
                z = f0_lin2(z);    z = nnx.silu(z); z = f0_ln2(z)
                z = f0_lin3(z);    z = nnx.silu(z); z = f0_ln3(z)
                bc0 = f0_out(z)

                # khat([phys, Rv]): lin1 → SiLU → lin2 → SiLU → linout → Softplus
                kh_lin1, kh_lin2, kh_out = self._kh_layers
                xk = jnp.concatenate([phys, Rv], axis=-1)
                zk = nnx.silu(kh_lin1(xk))
                zk = nnx.silu(kh_lin2(zk))
                k_hat = jax.nn.softplus(kh_out(zk))  # beta=1.0

                # resid(full x): Linear → SiLU → Linear  (no LN in your PyTorch)
                rs_lin1, rs_out = self._rs_layers
                r_hat = rs_out(nnx.silu(rs_lin1(x_file_norm)))

                # compose: BC convention (M_bol − M_band) ⇒ extinction lowers BC
                return bc0 + r_hat - Av * k_hat

            self._eval_core_v2 = _forward_v2
            self.eval = self.evalMLP_v2
    
    
        nnh5.close()

    def evalMLP(self,x):

        x_i = jnp.copy(jnp.asarray(x))        

        if self.normed:
            x_ii = jnp.zeros(x.shape,dtype=float)
            if len(x.shape) == 1:
                for ii,n_i in enumerate(self.norm_i):
                    mid = n_i[0]
                    std = n_i[1]
                    x_n = (x_i[ii]-mid)/std
                    x_ii = x_ii.at[ii].set(x_n)
            else:
                for ii,n_i in enumerate(self.norm_i):
                    mid = n_i[0]
                    std = n_i[1]
                    x_n = (x_i[:,ii]-mid)/std
                    x_ii = x_ii.at[:,ii].set(x_n)
        else:
            x_ii = x_i

        y = self.mlp(x_ii)

        if self.normed:
            y_i = jnp.zeros(y.shape,dtype=float)
            if len(x.shape) == 1:
                for ii,n_i in enumerate(self.norm_o):
                    mid = n_i[0]
                    std = n_i[1]
                    y_n = (y[ii]*std) + mid
                    y_i = y_i.at[ii].set(y_n)
            else:
                for ii,n_i in enumerate(self.norm_o):
                    mid = n_i[0]
                    std = n_i[1]
                    y_n = (y[:,ii]*std) + mid
                    y_i = y_i.at[:,ii].set(y_n)
        else:
            y_i = y

        return y_i        

    def evalMLP_v2(self, x):
        import jax.numpy as jnp
        x_i = jnp.copy(jnp.asarray(x))

        # track if caller gave a single example
        single = (x_i.ndim == 1)

        # 1) remap caller order → file label_i order (handles Teff→logt if needed)
        x_for_file = self._build_input_from_teff_order(x_i)

        # 2) normalize using self.norm_i
        if self.normed:
            x_ii = jnp.zeros(x_for_file.shape, dtype=float)
            if single:
                for ii, n_i in enumerate(self.norm_i):
                    mid, std = n_i[0], n_i[1]
                    x_ii = x_ii.at[ii].set((x_for_file[ii] - mid) / std)
            else:
                for ii, n_i in enumerate(self.norm_i):
                    mid, std = n_i[0], n_i[1]
                    x_ii = x_ii.at[:, ii].set((x_for_file[:, ii] - mid) / std)
        else:
            x_ii = x_for_file

        # 3) forward (returns shape (B, D_out) if we batched)
        y = self._eval_core_v2(x_ii)

        # ---- squeeze here for single-example path ----
        if single and y.ndim == 2 and y.shape[0] == 1:
            y = y[0]

        # 4) denormalize using self.norm_o
        if self.normed:
            y_i = jnp.zeros(y.shape, dtype=float)
            if single:
                for ii, n_i in enumerate(self.norm_o):
                    mid, std = n_i[0], n_i[1]
                    y_i = y_i.at[ii].set(y[ii] * std + mid)
            else:
                for ii, n_i in enumerate(self.norm_o):
                    mid, std = n_i[0], n_i[1]
                    y_i = y_i.at[:, ii].set(y[:, ii] * std + mid)
        else:
            y_i = y
        return y_i

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
