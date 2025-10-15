import torch
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

# a physics informed 2-head model, 1st head deals with stellar parameters, 2nd head
# deals with extinction, and then there is a residual head that accounts for non-linear
# behavior between the stellar SED and the extinction curve.

class MLP_v2(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out, d_phys=4, d_full=6):
        super().__init__()
        # stellar head f0: takes only the 4 physical inputs
        self.f0 = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(d_phys, H1)),
            ('af1', nn.SiLU()),
            ('ln1', nn.LayerNorm(H1)),
            ('lin2', nn.Linear(H1, H2)),
            ('af2', nn.SiLU()),
            ('ln2', nn.LayerNorm(H2)),
            ('d1', nn.Dropout(0.01)),
            ('lin3', nn.Linear(H2, H3)),
            ('af3', nn.SiLU()),
            ('ln3', nn.LayerNorm(H3)),
            ('linout', nn.Linear(H3, D_out)),
        ]))

        # # extinction lane khat: (phys + Rv) → per-band k_hat
        W = 96
        self.khat = nn.Sequential(OrderedDict([
            ('lin1',   nn.Linear(d_phys + 1, W)),  # same input as before: [phys, Rv]
            ('af1',    nn.SiLU()),
            ('lin2',   nn.Linear(W, W)),
            ('af2',    nn.SiLU()),
            ('linout', nn.Linear(W, D_out)),
            ('sp',     nn.Softplus(beta=1.0)),     # no params; JAX can ignore
        ]))
        # init linout small
        nn.init.zeros_(self.khat.linout.weight)
        nn.init.zeros_(self.khat.linout.bias)

        # small residual on the full 6-dim input
        W_resid = 32
        self.resid = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(d_full, W_resid)),      # 6
            ('af1', nn.SiLU()),
            ('lin2', nn.Linear(W_resid, D_out)),
        ]))

    def forward(self, x, return_khat: bool = False):
        # parse inputs
        x_phys = x[:, :4]      # [logt, logg, feh, afe]
        av     = x[:, 4]     # [Av]
        rv     = x[:, 5]     # [Rv]
        
        # compute 3 heads
        bc0    = self.f0(x_phys)
        k_hat  = self.khat(torch.cat([x_phys, rv[:,None]], dim=1))
        r_hat  = self.resid(x)

        # sum heads
        yhat = bc0 + (-av[:,None]) * k_hat + r_hat

        if return_khat:
            return yhat, k_hat
        else:
            return yhat


class KhatHead(nn.Module):
    """
    Extinction head: (phys, Rv) -> per-band k_hat
    - No physics prior (CCM/F99) baked in.
    - FiLM-modulates stellar features with Rv.
    - Softplus on output to keep k_hat >= 0 (good prior for Gaia).
    """
    def __init__(self,
                 d_phys: int,
                 D_out: int,
                 width: int = 128,
                 depth: int = 2,
                 use_film: bool = True,
                 dropout: float = 0.0,
                 softplus_eps: float = 1e-4):
        super().__init__()
        self.use_film = use_film
        self.softplus_eps = softplus_eps

        # (1) feature extractor on stellar parameters (phys)
        layers = [nn.Linear(d_phys, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        self.f_phys = nn.Sequential(*layers)

        # (2) conditioner on Rv -> FiLM params (gamma, beta)
        if use_film:
            self.f_rv = nn.Sequential(
                nn.Linear(1, width), nn.SiLU(),
                nn.Linear(width, 2 * width)  # -> [gamma | beta]
            )
        else:
            # fallback: just concatenate Rv to phys at input of a tiny block
            self.rv_cat = nn.Sequential(
                nn.Linear(width + 1, width), nn.SiLU()
            )

        # (3) band readout
        self.readout = nn.Linear(width, D_out)
        self.softplus = nn.Softplus(beta=1.0)

        # Conservative init (keeps early training stable)
        nn.init.kaiming_uniform_(self.readout.weight, nonlinearity='linear')
        nn.init.zeros_(self.readout.bias)

    def forward(self, phys: torch.Tensor, rv: torch.Tensor) -> torch.Tensor:
        """
        phys: (B, d_phys)
        rv  : (B,)        # scalar R_V per sample
        returns k_hat: (B, D_out)
        """
        h = self.f_phys(phys)  # (B, W)
        if self.use_film:
            gb = self.f_rv(rv[:, None])            # (B, 2W)
            gamma, beta = gb.chunk(2, dim=-1)
            h = gamma * h + beta                   # FiLM modulation by Rv
        else:
            h = self.rv_cat(torch.cat([h, rv[:, None]], dim=-1))
        k = self.readout(h)
        return self.softplus(k) + self.softplus_eps