"""NNmodels_new_v3.py

Neural network architectures for stellar spectral emulation.

Public API
----------
SpectralMLP_v3
    Physics-informed three-head MLP:
      f0     — stellar head (logt, logg, feh, afe [, vmic]) → log10 flux
      khat   — dust extinction head ([phys, Rv]) → k̂(λ) ≥ 0
      resid  — small non-linear correction on all inputs

    Supports an optional low-rank PCA basis so the network predicts K
    coefficients that are linearly decoded to L pixels, rather than
    predicting all L pixels directly (strongly recommended for L > 2000).

Design notes
------------
- All outputs are in log10 flux space.
- Extinction is applied as  y = base_log + (-0.4 * Av) * k̂(λ) * g_ext
  which is the correct log10 form of  f_ext = f_0 * 10^{-0.4 Av k(λ)}.
- The residual lane is capped with tanh-rescaling (smooth, gradient-safe)
  rather than a hard clamp.
- Learnable sigmoid gates (ext_gate, res_gate) allow each lane to be
  softly enabled/disabled during phased training without toggling requires_grad.
- freeze_f0() is provided for the phase-2 warm-in period; it is a one-shot
  call, not a per-epoch toggle, to avoid stale optimizer moments.
"""
from __future__ import annotations

import torch
from torch import nn
from collections import OrderedDict
from typing import Optional


# -----------------------------------------------------------------------
# Fixed linear decoder (basis projection)
# -----------------------------------------------------------------------
class FixedMatMul(nn.Module):
    """Fixed (non-trainable) linear projection: z (B,K) → y (B,L).

    Registered as a buffer so it moves to the correct device with .to()
    and is saved/loaded with the state dict, but receives no gradients.
    """

    def __init__(self, B: torch.Tensor):
        super().__init__()
        assert B.ndim == 2, "B must be a 2-D tensor of shape (K, L)."
        self.register_buffer("B", B.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, K),  B: (K, L)  →  out: (batch, L)
        return x @ self.B


# -----------------------------------------------------------------------
# Main architecture
# -----------------------------------------------------------------------
class SpectralMLP_v3(nn.Module):
    """Physics-informed MLP for stellar spectral emulation.

    Parameters
    ----------
    d_phys : int
        Number of purely stellar inputs (e.g. 4 for logt/logg/feh/afe, or 5
        if vmic is free). These are fed to f0 and khat.
    d_full : int
        Total number of inputs including av and rv if present (d_phys + 0/2).
    L : int
        Number of output spectral pixels.
    H1, H2, H3 : int
        Hidden widths of the main stellar head f0.
    W_k : int
        Hidden width of the extinction head khat.
    W_resid : int
        Hidden width of the residual correction head.
    basis_B : Tensor of shape (K, L), optional
        PCA basis matrix. If provided, f0 predicts K coefficients decoded by B.
        Strongly recommended for L > 2000.
    mu_log : Tensor of shape (L,) or (1, L), optional
        Mean log10 spectrum used as a bias after f0 when using the basis head.
        If None and basis is used, a trainable bias is created instead.
    include_extinction : bool, default True
        Enable the khat extinction lane.
    include_resid : bool, default True
        Enable the non-linear residual correction lane.
    inputs_have_avrv : bool, default True
        Whether the input vector contains av and rv at positions [d_phys] and
        [d_phys+1]. Set False for phase-1 (intrinsic-only) training.
    ext_gate_init : float, default 1.0
        Pre-sigmoid initial value for the extinction gate (sigmoid(1) ≈ 0.73).
    resid_gate_init : float, default -2.0
        Pre-sigmoid initial value for the residual gate (sigmoid(-2) ≈ 0.12),
        keeping the residual lane near zero at the start of phase 3.
    max_khat : float, default 10.0
        Hard upper cap on k̂(λ) applied after Softplus.
    max_resid_dex : float, default 0.5
        Scale for tanh-based soft capping of the residual: 
        r_out = max_resid_dex * tanh(r_raw / max_resid_dex).
        This is smooth and gradient-safe, unlike a hard clamp.
    """

    def __init__(
        self,
        d_phys: int,
        d_full: int,
        L: int,
        H1: int = 512,
        H2: int = 512,
        H3: int = 512,
        W_k: int = 128,
        W_resid: int = 256,
        basis_B: Optional[torch.Tensor] = None,
        mu_log: Optional[torch.Tensor] = None,
        include_extinction: bool = True,
        include_resid: bool = True,
        inputs_have_avrv: bool = True,
        ext_gate_init: float = 1.0,
        resid_gate_init: float = -2.0,
        max_khat: float = 10.0,
        max_resid_dex: float = 0.5,
    ):
        super().__init__()
        self.d_phys             = d_phys
        self.d_full             = d_full
        self.L                  = L
        self.include_extinction = include_extinction
        self.include_resid      = include_resid
        self.inputs_have_avrv   = inputs_have_avrv
        self.has_basis          = basis_B is not None
        self.max_khat           = float(max_khat)
        self.max_resid_dex      = float(max_resid_dex)

        # ----------------------------------------------------------------
        # f0: stellar head  phys → log10 flux  (or → K coefficients)
        # ----------------------------------------------------------------
        f0_body = [
            ("lin1", nn.Linear(d_phys, H1)),
            ("af1",  nn.SiLU()),
            ("ln1",  nn.LayerNorm(H1)),
            ("lin2", nn.Linear(H1, H2)),
            ("af2",  nn.SiLU()),
            ("ln2",  nn.LayerNorm(H2)),
            ("lin3", nn.Linear(H2, H3)),
            ("af3",  nn.SiLU()),
            ("ln3",  nn.LayerNorm(H3)),
        ]

        if not self.has_basis:
            # Direct pixel prediction
            f0_body.append(("linout", nn.Linear(H3, L)))
            self.f0      = nn.Sequential(OrderedDict(f0_body))
            self.mu_log  = None
            self.log_bias = None
        else:
            K = basis_B.shape[0]
            f0_body.append(("toK", nn.Linear(H3, K)))
            f0_body.append(("B",   FixedMatMul(basis_B)))
            self.f0 = nn.Sequential(OrderedDict(f0_body))
            # Small init on the coefficient layer so we start near μ
            with torch.no_grad():
                nn.init.normal_(self.f0.toK.weight, mean=0.0, std=1e-2)
                nn.init.zeros_(self.f0.toK.bias)
            # Mean log spectrum as fixed bias or trainable parameter
            if mu_log is not None and isinstance(mu_log, torch.Tensor):
                self.register_buffer(
                    "mu_log", mu_log.reshape(1, L).contiguous().float()
                )
                self.log_bias = None
            else:
                self.mu_log   = None
                self.log_bias = nn.Parameter(torch.zeros(1, L))

        # ----------------------------------------------------------------
        # khat: extinction head  [phys, Rv] → k̂(λ) ≥ 0
        # ----------------------------------------------------------------
        self.khat = nn.Sequential(OrderedDict([
            ("lin1",   nn.Linear(d_phys + 1, W_k)),
            ("af1",    nn.SiLU()),
            ("lin2",   nn.Linear(W_k, W_k)),
            ("af2",    nn.SiLU()),
            ("linout", nn.Linear(W_k, L)),
            ("sp",     nn.Softplus(beta=1.0)),   # enforces k̂ ≥ 0
        ]))
        # Zero-init output so k̂ starts near 0 (extinction starts inactive)
        nn.init.zeros_(self.khat.linout.weight)
        nn.init.zeros_(self.khat.linout.bias)

        # ----------------------------------------------------------------
        # resid: non-linear correction  [all inputs] → Δ log flux
        # Wider than v1/v2 to capture extinction–stellar cross-terms.
        # Two hidden layers for extra capacity.
        # ----------------------------------------------------------------
        self.resid = nn.Sequential(OrderedDict([
            ("lin1", nn.Linear(d_full, W_resid)),
            ("af1",  nn.SiLU()),
            ("lin2", nn.Linear(W_resid, W_resid)),
            ("af2",  nn.SiLU()),
            ("linout", nn.Linear(W_resid, L)),
        ]))
        # Zero-init so residual lane starts silent
        nn.init.zeros_(self.resid.linout.weight)
        nn.init.zeros_(self.resid.linout.bias)

        # ----------------------------------------------------------------
        # Learnable gates (scalar sigmoid in forward)
        # Stored as length-1 Parameters to avoid 0-D tensor issues with h5py.
        # ----------------------------------------------------------------
        self.ext_gate  = nn.Parameter(torch.ones(1) * float(ext_gate_init))
        self.res_gate  = nn.Parameter(torch.ones(1) * float(resid_gate_init))

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_khat: bool = False,
    ):
        """
        Parameters
        ----------
        x : Tensor, shape (B, d_full)
            Input feature vector. Physical params first, then av, rv.
        return_khat : bool
            If True, return (y_log, k_hat) instead of just y_log.

        Returns
        -------
        y_log : Tensor, shape (B, L)
            Predicted log10 flux.
        k_hat : Tensor, shape (B, L)   [only if return_khat=True]
            Predicted normalised extinction curve.
        """
        x_phys = x[:, : self.d_phys]

        if self.inputs_have_avrv and x.size(1) >= self.d_phys + 2:
            av = x[:, self.d_phys]
            rv = x[:, self.d_phys + 1]
        else:
            av = x.new_zeros(x.size(0))
            rv = x.new_full((x.size(0),), 3.1)

        # ---- stellar head ----
        base_log = self.f0(x_phys)
        if self.mu_log is not None:
            base_log = base_log + self.mu_log
        elif self.log_bias is not None:
            base_log = base_log + self.log_bias

        # ---- gates ----
        g_ext = torch.sigmoid(self.ext_gate)   # ∈ (0, 1)
        g_res = torch.sigmoid(self.res_gate)

        # ---- extinction lane ----
        if self.include_extinction:
            k_hat = self.khat(torch.cat([x_phys, rv[:, None]], dim=1))   # (B, L)
            k_hat = torch.clamp(k_hat, max=self.max_khat)
            ext_term = (-0.4 * av[:, None]) * k_hat * g_ext
        else:
            k_hat    = x.new_zeros(x.size(0), self.L)
            ext_term = k_hat

        # ---- residual lane  (tanh soft cap, gradient-safe) ----
        if self.include_resid:
            r_in = (x if self.inputs_have_avrv
                    else torch.cat([x_phys, av[:, None], rv[:, None]], dim=1))
            r_raw = self.resid(r_in)
            # tanh rescaling: smooth, bounded by ±max_resid_dex, no gradient kill
            r_hat = self.max_resid_dex * torch.tanh(r_raw / self.max_resid_dex)
            r_hat = r_hat * g_res
        else:
            r_hat = x.new_zeros(x.size(0), self.L)

        y_log = base_log + ext_term + r_hat
        return (y_log, k_hat) if return_khat else y_log

    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------
    @torch.no_grad()
    def freeze_f0(self, freeze: bool = True) -> "SpectralMLP_v3":
        """Freeze or unfreeze the stellar head f0.

        Call once before the training loop starts (not per-epoch) to avoid
        stale AdamW moment buffers accumulating while gradients are disabled.
        The gates remain trainable regardless.
        """
        for p in self.f0.parameters():
            p.requires_grad = not freeze
        self.ext_gate.requires_grad = True
        self.res_gate.requires_grad = True
        return self

    @torch.no_grad()
    def freeze_extinction(self, freeze: bool = True) -> "SpectralMLP_v3":
        """Freeze or unfreeze the khat extinction head."""
        for p in self.khat.parameters():
            p.requires_grad = not freeze
        return self

    def coefficients(self, x_phys: torch.Tensor) -> Optional[torch.Tensor]:
        """Return PCA coefficients z of shape (B, K) for given stellar inputs.

        Only available when a basis was provided at construction time.
        Returns None otherwise.

        Parameters
        ----------
        x_phys : Tensor, shape (B, d_phys)
            Stellar parameter inputs (not the full input vector).
        """
        if not self.has_basis:
            return None
        # Run only up to (but not including) the FixedMatMul projection
        # so we recover the raw coefficient vector z.
        z = x_phys
        for name, layer in self.f0.named_children():
            if name == "B":          # FixedMatMul — stop here
                break
            z = layer(z)
        return z

    def decode_coefficients(self, z: torch.Tensor) -> torch.Tensor:
        """Decode PCA coefficients z (B, K) back to log10 flux (B, L).

        Only meaningful when a basis was provided.
        """
        if not self.has_basis:
            raise RuntimeError("No basis provided; cannot decode coefficients.")
        y = self.f0.B(z)             # FixedMatMul: (B,K)@(K,L) → (B,L)
        if self.mu_log is not None:
            y = y + self.mu_log
        elif self.log_bias is not None:
            y = y + self.log_bias
        return y

    def n_params(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )