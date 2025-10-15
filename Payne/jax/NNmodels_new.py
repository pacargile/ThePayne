from torch import nn
from collections import OrderedDict


# linear feed-foward model with simple activation functions
class MLP_v0(nn.Module):  
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(MLP_v0, self).__init__()


        self.mlp = nn.Sequential(OrderedDict([
            ('lin1',nn.Linear(D_in, H1)),
            ('ln1',nn.LayerNorm(H1)),
            ('af1',nn.SiLU()),  
            ('lin2',nn.Linear(H1, H2)),
            ('ln2',nn.LayerNorm(H2)),
            ('af2',nn.SiLU()),
            ('lin3',nn.Linear(H2, H3)),
            ('ln3',nn.LayerNorm(H3)),
            ('af3',nn.SiLU()),
            ('d1',nn.Dropout(0.3)),
            ('lin4',nn.Linear(H3, H3)),
            ('ln4',nn.LayerNorm(H3)),
            ('af4',nn.SiLU()),
            ('lin5', nn.Linear(H3, H3)),
            ('ln5', nn.LayerNorm(H3)),
            ('af5', nn.SiLU()),
            ('lin6', nn.Linear(H3, D_out))
        ]))

    def forward(self, x):
        y_i = self.mlp(x)
        return y_i     
    
    
# linear feed-foward model with simple activation functions
class MLP_v1(nn.Module):  
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(MLP_v1, self).__init__()

        self.mlp = nn.Sequential(OrderedDict([
            ('lin1',nn.Linear(D_in, H1)),
            ('ln1',nn.LayerNorm(H1)),
            ('af1',nn.SiLU()),  
            ('lin2',nn.Linear(H1, H2)),
            ('ln2',nn.LayerNorm(H2)),
            ('af2',nn.SiLU()),
            ('d1',nn.Dropout(0.01)),
            ('lin3',nn.Linear(H2, H3)),
            ('ln3',nn.LayerNorm(H3)),
            ('af3',nn.SiLU()),
            ('linout', nn.Linear(H3, D_out))
        ]))

    def forward(self, x):
        y_i = self.mlp(x)
        return y_i     


        if (nntype == 'MLP_v2'):
            # ---------- read weights ----------
            # f0 head
            b1   = jnp.array(nnh5['model/f0.lin1.bias'][()]);     W1   = jnp.array(nnh5['model/f0.lin1.weight'][()])
            b2   = jnp.array(nnh5['model/f0.lin2.bias'][()]);     W2   = jnp.array(nnh5['model/f0.lin2.weight'][()])
            b3   = jnp.array(nnh5['model/f0.lin3.bias'][()]);     W3   = jnp.array(nnh5['model/f0.lin3.weight'][()])
            bout = jnp.array(nnh5['model/f0.linout.bias'][()]);   Wout = jnp.array(nnh5['model/f0.linout.weight'][()])

            ln1b = jnp.array(nnh5['model/f0.ln1.bias'][()]);      ln1s = jnp.array(nnh5['model/f0.ln1.weight'][()])
            ln2b = jnp.array(nnh5['model/f0.ln2.bias'][()]);      ln2s = jnp.array(nnh5['model/f0.ln2.weight'][()])
            ln3b = jnp.array(nnh5['model/f0.ln3.bias'][()]);      ln3s = jnp.array(nnh5['model/f0.ln3.weight'][()])

            # khat head
            kb1  = jnp.array(nnh5['model/khat.khat_lin1.bias'][()]);  kW1  = jnp.array(nnh5['model/khat.khat_lin1.weight'][()])
            klnb = jnp.array(nnh5['model/khat.khat_ln1.bias'][()]);   klns = jnp.array(nnh5['model/khat.khat_ln1.weight'][()])
            kb2  = jnp.array(nnh5['model/khat.khat_lin2.bias'][()]);  kW2  = jnp.array(nnh5['model/khat.khat_lin2.weight'][()])

            # transpose PyTorch [out,in] -> JAX [in,out]
            W1, W2, W3, Wout = (W1.T, W2.T, W3.T, Wout.T)
            kW1, kW2 = (kW1.T, kW2.T)

            # ---------- build layers ----------
            # f0 path
            f0_lin1 = nnx.Linear(in_features=W1.shape[0], out_features=W1.shape[1], rngs=nnx.Rngs(0))
            f0_lin1.kernel = nnx.Param(value=W1);   f0_lin1.bias = nnx.Param(value=b1)
            f0_ln1  = nnx.LayerNorm(num_features=f0_lin1.bias.shape[0], rngs=nnx.Rngs(0))
            f0_ln1.bias = nnx.Param(value=ln1b);    f0_ln1.scale = nnx.Param(value=ln1s)

            f0_lin2 = nnx.Linear(in_features=W2.shape[0], out_features=W2.shape[1], rngs=nnx.Rngs(0))
            f0_lin2.kernel = nnx.Param(value=W2);   f0_lin2.bias = nnx.Param(value=b2)
            f0_ln2  = nnx.LayerNorm(num_features=f0_lin2.bias.shape[0], rngs=nnx.Rngs(0))
            f0_ln2.bias = nnx.Param(value=ln2b);    f0_ln2.scale = nnx.Param(value=ln2s)

            f0_lin3 = nnx.Linear(in_features=W3.shape[0], out_features=W3.shape[1], rngs=nnx.Rngs(0))
            f0_lin3.kernel = nnx.Param(value=W3);   f0_lin3.bias = nnx.Param(value=b3)
            f0_ln3  = nnx.LayerNorm(num_features=f0_lin3.bias.shape[0], rngs=nnx.Rngs(0))
            f0_ln3.bias = nnx.Param(value=ln3b);    f0_ln3.scale = nnx.Param(value=ln3s)

            f0_out  = nnx.Linear(in_features=Wout.shape[0], out_features=Wout.shape[1], rngs=nnx.Rngs(0))
            f0_out.kernel = nnx.Param(value=Wout);  f0_out.bias = nnx.Param(value=bout)

            # khat path
            kh_lin1 = nnx.Linear(in_features=kW1.shape[0], out_features=kW1.shape[1], rngs=nnx.Rngs(0))
            kh_lin1.kernel = nnx.Param(value=kW1);  kh_lin1.bias = nnx.Param(value=kb1)
            kh_ln1  = nnx.LayerNorm(num_features=kh_lin1.bias.shape[0], rngs=nnx.Rngs(0))
            kh_ln1.bias = nnx.Param(value=klnb);    kh_ln1.scale = nnx.Param(value=klns)

            kh_out  = nnx.Linear(in_features=kW2.shape[0], out_features=kW2.shape[1], rngs=nnx.Rngs(0))
            kh_out.kernel = nnx.Param(value=kW2);   kh_out.bias = nnx.Param(value=kb2)

            # stash handles we’ll use in the forward
            self._f0_layers = (f0_lin1, f0_ln1, f0_lin2, f0_ln2, f0_lin3, f0_ln3, f0_out)
            self._kh_layers = (kh_lin1, kh_ln1, kh_out)

            # v2 forward
            def _forward_v2(x_in):
                # x_in expected shape (..., 6) in the canonical order:
                # [Teff, logg, feh, afe, Av, Rv]
                # We rely on your training convention. If your label_i ever changes,
                # swap to index-by-name here.
                if x_in.ndim == 1:
                    x_in = x_in[None, :]
                phys = x_in[:, :4]
                Av   = x_in[:, 4:5]
                Rv   = x_in[:, 5:6]

                # normalize input (done in eval wrapper; we assume already normalized here)
                # ----- f0(phys) -----
                f0_lin1, f0_ln1, f0_lin2, f0_ln2, f0_lin3, f0_ln3, f0_out = self._f0_layers
                z = f0_lin1(phys); z = f0_ln1(z); z = nnx.silu(z)
                z = f0_lin2(z);    z = f0_ln2(z); z = nnx.silu(z)
                # dropout during training was p=0.01, but predictor is eval-only -> skip
                z = f0_lin3(z);    z = f0_ln3(z); z = nnx.silu(z)
                m0 = f0_out(z)

                # ----- khat([phys, Rv]) -----
                kh_lin1, kh_ln1, kh_out = self._kh_layers
                xk = jnp.concatenate([phys, Rv], axis=-1)
                zk = kh_lin1(xk); zk = kh_ln1(zk); zk = nnx.silu(zk)
                k_hat = kh_out(zk)

                # compose extinction
                m = m0 + Av * k_hat
                return m

            self._eval_core_v2 = _forward_v2
            self.eval = self.evalMLP_v2