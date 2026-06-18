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



#### Spectral MLP with physics-informed architecture

class FixedMatMul(nn.Module):
    def __init__(self, B: torch.Tensor):
        super().__init__()
        assert B.ndim == 2, "B must be (K, L)"
        self.register_buffer("B", B.float())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.B


class SpectralMLP_v1(nn.Module):
    def __init__(
        self,
        d_phys: int,
        d_full: int,
        L: int,
        H1: int = 512, H2: int = 512, H3: int = 512,
        W_k: int = 128, 
        W_resid: int = 64,
        basis_B: Optional[torch.Tensor] = None,   # (K, L) or None
        mu_log: Optional[torch.Tensor] = None,    # (L,) or (1,L) or None
        include_extinction: bool = True,
        include_resid: bool = True,
        inputs_have_avrv: bool = True,
        ext_gate_init: float = 1.0,               # sigmoid(ext_gate) scales extinction term
        resid_gate_init: float = 0.0,             # start residual lane near 0
        max_khat: float = 10.0,                   # cap for k_hat (dex/Av scale)
        max_resid_dex: float = 0.5,               # cap |residual| in dex        
    ):
        super().__init__()
        self.d_phys = d_phys
        self.d_full = d_full
        self.L = L
        self.include_extinction = include_extinction
        self.include_resid = include_resid
        self.inputs_have_avrv = inputs_have_avrv
        self.has_basis = basis_B is not None
        self.max_khat = float(max_khat)
        self.max_resid_dex = float(max_resid_dex)
        
        # ----- f0: stellar head -----
        f0_layers = [
            ('lin1',   nn.Linear(d_phys, H1)),
            ('af1',    nn.SiLU()),
            ('ln1',    nn.LayerNorm(H1)),
            ('lin2',   nn.Linear(H1, H2)),
            ('af2',    nn.SiLU()),
            ('ln2',    nn.LayerNorm(H2)),
            ('lin3',   nn.Linear(H2, H3)),
            ('af3',    nn.SiLU()),
            ('ln3',    nn.LayerNorm(H3)),
        ]

        if basis_B is None:
            f0_layers += [('linout', nn.Linear(H3, L))]
            self.f0 = nn.Sequential(OrderedDict(f0_layers))
            self.mu_log = None
            self.log_bias = None
        else:
            K = basis_B.shape[0]
            f0_layers += [('toK', nn.Linear(H3, K)), ('B', FixedMatMul(basis_B))]
            self.f0 = nn.Sequential(OrderedDict(f0_layers))
            # small init for toK so we start near μ but with gradient room
            with torch.no_grad():
                nn.init.normal_(self.f0.toK.weight, mean=0.0, std=1e-2)
                nn.init.zeros_(self.f0.toK.bias)
            if (mu_log is not None) and isinstance(mu_log, torch.Tensor):
                self.register_buffer("mu_log", mu_log.reshape(1, L).contiguous().float())
                self.register_parameter("log_bias", None)
            else:
                self.mu_log = None
                self.log_bias = nn.Parameter(torch.zeros(1, L))

        # ----- khat: (phys + Rv) → k_hat(λ) ≥ 0 -----
        self.khat = nn.Sequential(OrderedDict([
            ('lin1',   nn.Linear(d_phys + 1, W_k)),
            ('af1',    nn.SiLU()),
            ('lin2',   nn.Linear(W_k, W_k)),
            ('af2',    nn.SiLU()),
            ('linout', nn.Linear(W_k, L)),
            ('sp',     nn.Softplus(beta=1.0)),
        ]))
        nn.init.zeros_(self.khat.linout.weight)
        nn.init.zeros_(self.khat.linout.bias)

        # ----- resid head -----
        self.resid = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(d_full, W_resid)),
            ('af1',  nn.SiLU()),
            ('lin2', nn.Linear(W_resid, L)),
        ]))

        # ----- learnable gates (sigmoid in forward) -----
        # Use 1D length-1 params to avoid 0-D (scalar) tensors that upset h5py compression.
        self.ext_gate  = nn.Parameter(torch.ones(1) * float(ext_gate_init))
        self.res_gate  = nn.Parameter(torch.ones(1) * float(resid_gate_init))

    def forward(self, x: torch.Tensor, return_khat: bool = False):
        x_phys = x[:, :self.d_phys]
        if self.inputs_have_avrv and x.size(1) >= self.d_phys + 2:
            av = x[:, self.d_phys]
            rv = x[:, self.d_phys + 1]
        else:
            B = x.size(0)
            av = x.new_zeros(B)
            rv = x.new_full((B,), 3.1)

        base_log = self.f0(x_phys)
        if self.mu_log is not None:
            base_log = base_log + self.mu_log
        elif getattr(self, "log_bias", None) is not None:
            base_log = base_log + self.log_bias

        # gates ∈ (0,1]; keep differentiable and numerically tame
        g_ext  = torch.sigmoid(self.ext_gate)
        g_res  = torch.sigmoid(self.res_gate)

        if self.include_extinction:
            k_hat = self.khat(torch.cat([x_phys, rv[:, None]], dim=1))           # (B,L) ≥ 0 after Softplus
            if self.max_khat is not None:
                k_hat = torch.clamp(k_hat, max=self.max_khat)
            ext_term = (-0.4 * av[:, None]) * k_hat * g_ext
        else:
            k_hat = x.new_zeros((x.size(0), self.L))
            ext_term = k_hat

        if self.include_resid:
            r_in = x if self.inputs_have_avrv else torch.cat([x_phys, av[:, None], rv[:, None]], dim=1)
            r_hat = self.resid(r_in)
            # cap the residual in dex so the lane cannot explode
            if self.max_resid_dex is not None:
                r_hat = torch.clamp(r_hat, min=-self.max_resid_dex, max=self.max_resid_dex)
            r_hat = r_hat * g_res
        else:
            r_hat = x.new_zeros((x.size(0), self.L))

        y_log = base_log + ext_term + r_hat
        return (y_log, k_hat) if return_khat else y_log

    # ---------- utilities ----------
    @torch.no_grad()
    def freeze_f0(self, freeze: bool = True):
        """Convenience toggle to freeze/unfreeze the main stellar head."""
        for p in self.f0.parameters():
            p.requires_grad = (not freeze)
        # keep gates trainable
        for p in (self.ext_gate, self.res_gate):
            p.requires_grad = True
        return self

    def coefficients(self, y_log: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return low-rank coefficients z (B,K) if a basis is present, else None.
        y_log must be in log10 flux space.
        """
        if not (hasattr(self, "f0") and hasattr(self.f0, "B") and hasattr(self.f0.B, "B")):
            return None
        mu = self.mu_log if (hasattr(self, "mu_log") and isinstance(self.mu_log, torch.Tensor)) else 0.0
        Bt = self.f0.B.B.t()  # (L,K)
        return (y_log - mu) @ Bt