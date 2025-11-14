import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np, math, time, os, h5py
from datetime import datetime
from collections import OrderedDict

# your utils
from ..utils import readKorg 
from ..utils.readKorg import XYFromFlat  # same wrapper
from ..utils.io_h5 import save_state_dict_to_h5, load_state_dict_from_h5, save_labels_norms_to_h5, save_meta_to_h5

# the model
from .NNmodels_new import SpectralMLP_v1

# for plotting
import matplotlib
matplotlib.use('AGG')         # headless
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step")

# ---- device setup (copy from your photometric file) ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def _unwrap(m): return getattr(m, "_orig_mod", m)
def _nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
def _fmt(n):
    return f"{n/1e6:.2f} M" if n>=1e6 else f"{n/1e3:.2f} K" if n>=1e3 else str(n)

def _seed_everything(seed: int = 1337):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class TrainSpec:
    """
    Spectral trainer modeled after your photometric trainer.
    Trains in **log10 flux** space.

    Typical phases you might run:
      1) Intrinsic only (no extinction, no resid): include_extinction=False, include_resid=False,
         dataset extinction_mode='none', inputs_have_avrv=False, label_i excludes Av/Rv
      2) Add extinction (still no resid): include_extinction=True,
         dataset extinction_mode='sample', inputs_have_avrv=True, label_i includes Av/Rv
      3) Add resid: include_resid=True
    """
    def __init__(self, **cfg):
        print(f'... Start Spectral Training at {datetime.now()}')

        # --- core config (mirrors photometric) ---
        self.split_seed   = cfg.get('split_seed', 1337)
        self.logplot      = cfg.get('logplot', True)

        self.trainper     = cfg.get('trainper', 0.9)
        self.numepochs    = cfg.get('numepochs', 10000)
        self.batchsize    = cfg.get('batchsize', 64)      # spectra can be large; start modest
        self.lr           = cfg.get('lr', 1e-3)
        self.plotevery    = cfg.get('plotevery', 20)
        self.plotdir      = cfg.get('plotdir', './plots/')
        os.makedirs(self.plotdir, exist_ok=True)

        # scheduler / warmup / cosine
        self.warmup_epochs = cfg.get('warmup_epochs', max(1, int(0.05 * cfg.get('numepochs', 10000))))
        self.eta_min       = cfg.get('eta_min', 1e-5)   # cosine floor

        # gradient accumulation (stability with smaller step noise)
        self.grad_accum_steps = int(cfg.get('grad_accum_steps', 1))  # e.g., 8 to emulate large batch

        # per-pixel weighting in log space to balance bright/faint regions
        # when True we compute σ_log(λ) on the train subset and weight by 1/(σ^2 + eps)
        self.weight_by_logvar = bool(cfg.get('weight_by_logvar', True))
        self.weight_eps = float(cfg.get('weight_eps', 1e-6))

        # fractional-error term in linear flux
        self.use_frac_loss = bool(cfg.get("use_frac_loss", True))  # turn on/off
        self.frac_alpha    = float(cfg.get("frac_alpha", 0.5))     # mix with log loss
        self.frac_clip_dex   = float(cfg.get("frac_clip_dex", 4.0))  # clip residuals > this dex in frac loss

        # architecture
        self.H1           = cfg.get('H1', 512)
        self.H2           = cfg.get('H2', 512)
        self.H3           = cfg.get('H3', 512)
        self.W_k          = cfg.get('W_k', 96)
        self.W_resid      = cfg.get('W_resid', 64)

        # model toggles
        self.include_extinction = cfg.get('include_extinction', False)
        self.include_resid      = cfg.get('include_resid', False)
        self.inputs_have_avrv   = cfg.get('inputs_have_avrv', False)

        # dataset knobs (ReadSpec)
        self.modpath      = cfg.get('modpath', './grid/h5/')
        self.wave_range   = cfg.get('wave_range', (4000.0, 10000.0))
        self.dlambda      = cfg.get('dlambda', None)
        self.R            = cfg.get('R', 100.0)
        self.px_per_resel = cfg.get('pixels_per_resel', 3.0)
        self.rebin_mode   = cfg.get('rebin_mode', 'interp')
        self.norm_outputs = False  # we train on raw flux, so keep False
        self.ext_mode     = cfg.get('extinction_mode', 'none')  # 'none'|'sample'|'fixed'|'grid'
        self.fixed_av     = cfg.get('fixed_av', 0.0)
        self.fixed_rv     = cfg.get('fixed_rv', 3.1)

        # inputs/outputs (labels)
        # d_phys typically 5: [logt,logg,feh,afe,vmic]
        self.label_i      = cfg.get('label_i',
                                    ['logt','logg','feh','afe','vmic'] + (['av','rv'] if self.inputs_have_avrv else []))

        # label_o are wavelength labels from the dataset; we’ll populate after anchoring dataset
        self.label_o      = None

        # ranges (used by ReadSpec if provided)
        self.parrange = cfg.get('parrange', {
            'logt': [3.0, 4.7],     # ~1e3 K to 5e4 K; adjust as needed
            'logg': [-2.0, 6.0],
            'feh':  [-5.0, 1.0],
            'afe':  [-0.2, 0.6],
            'av':   [0.0, 20.0],
            'rv':   [2.0, 6.0],
        })

        # files
        self.restartfile  = cfg.get('restartfile', None)
        self.outfilename  = cfg.get('output', 'TRAIN_SPEC_OUT.h5')
        # if True, interpret numepochs as the *total* target epochs and
        # resume from epochs_trained stored in restartfile's meta attrs
        self.resume_from_restart = cfg.get('resume_from_restart', False)
        
        # loader workers
        self.num_workers  = cfg.get('num_workers', 0)

        # ES / LR schedule
        self.early_stopping           = cfg.get('early_stopping', True)
        self.early_stopping_patience  = cfg.get('early_stopping_patience', 100)
        self.early_stopping_min_delta = cfg.get('early_stopping_min_delta', 1e-5)

    def _build_datasets_global_split(self):
        """
        Build train/valid ReadSpec using one global, seeded permutation over the
        entire dataset (after any parrange filtering and wavelength/resampling).
        """
        # 1) Anchor pass: load with split=None so ReadSpec will construct its
        #    internal split; we'll *union* those indices to recover the full set.
        anchor = readKorg.ReadSpec(
            modpath=self.modpath,
            wave_range=self.wave_range,
            dlambda=self.dlambda,
            R=self.R,
            pixels_per_resel=self.px_per_resel,
            rebin_mode=self.rebin_mode,
            norm=self.norm_outputs,
            use_norm_from_h5=True,
            returntorch=True,
            type='train',           # arbitrary; only used to form base_block internally
            trainpercentage=1.0,    # load all rows; we'll control split explicitly next
            parrange=self.parrange,
            label_i=self.label_i,
            extinction_mode=self.ext_mode,
            fixed_av=self.fixed_av,
            fixed_rv=self.fixed_rv,
            split_seed=self.split_seed,
            split=None,
        )

        # 2) Recover full index universe from anchor's internal split
        si = anchor.split_indices
        all_idx = np.concatenate([si["train"], si["valid"], si["test"]]).astype(int)

        # 3) Global permutation + cut
        rng = np.random.RandomState(self.split_seed)
        perm = rng.permutation(all_idx)
        cut = int(self.trainper * perm.size)
        train_idx = np.sort(perm[:cut])
        valid_idx = np.sort(perm[cut:])
        test_idx  = np.array([], dtype=int)   # no holdout test by default

        split_dict = {"train": train_idx, "valid": valid_idx, "test": test_idx}

        # 4) Build actual train/valid datasets with explicit split dict
        ds_train = readKorg.ReadSpec(
            modpath=self.modpath,
            wave_range=self.wave_range,
            dlambda=self.dlambda,
            R=self.R,
            pixels_per_resel=self.px_per_resel,
            rebin_mode=self.rebin_mode,
            norm=self.norm_outputs,
            use_norm_from_h5=True,
            returntorch=True,
            type='train',
            trainpercentage=1.0,           # irrelevant when 'split' is provided
            parrange=self.parrange,
            label_i=self.label_i,
            extinction_mode=self.ext_mode,
            fixed_av=self.fixed_av,
            fixed_rv=self.fixed_rv,
            split_seed=self.split_seed,
            split=split_dict,
        )
        ds_valid = readKorg.ReadSpec(
            modpath=self.modpath,
            wave_range=self.wave_range,
            dlambda=self.dlambda,
            R=self.R,
            pixels_per_resel=self.px_per_resel,
            rebin_mode=self.rebin_mode,
            norm=self.norm_outputs,
            use_norm_from_h5=True,
            returntorch=True,
            type='valid',
            trainpercentage=1.0,           # irrelevant when 'split' is provided
            parrange=self.parrange,
            label_i=self.label_i,
            extinction_mode=('fixed' if self.ext_mode != 'none' else 'none'),
            fixed_av=self.fixed_av,
            fixed_rv=self.fixed_rv,
            split_seed=self.split_seed,
            split=split_dict,
        )

        self.label_o = list(ds_train.label_o)
        return ds_train, ds_valid, split_dict
    
    def _maybe_build_basis(self, ds_train, K=None):
        """
        Optional PCA basis in **log10 flux** on the training split.
        Returns (B, mu_log, sd_log) where:
          - B is (K, L) or None
          - mu_log is (L,) mean log10 spectrum (or None if K is None)
          - sd_log is (L,) std log10 spectrum (or None if K is None)
      """
        if K is None: 
            # still compute mean/std for logging
            K = 0
            
        # grab a subset to form PCA (cap to keep memory sane)
        idx = torch.arange(len(ds_train))
        # collect X -> (N, L) log flux
        L = len(ds_train.label_o)
        cap = min(20000, len(idx))
        samp = idx[:cap]
        X = []
        with torch.no_grad():
            for ii in samp:
                flat = ds_train[ii]
                y = flat[len(self.label_i):]  # flux
                y = torch.log10(torch.clamp(y, min=1e-12))
                X.append(y.unsqueeze(0))
        X = torch.cat(X, dim=0)  # (cap, L)

        # mean/var in log space (for loss weighting)
        mu_log = X.mean(dim=0, keepdim=True)     # (1, L)
        sd_log = X.std(dim=0, keepdim=True, unbiased=False)  # (1, L)
        Xm = X - mu_log

        # economy SVD only if K>0
        if int(K) > 0:
           U, S, Vt = torch.linalg.svd(Xm, full_matrices=False)
           K_req = int(K)
           L = Vt.shape[1]
           K_eff = min(K_req, L)
           if K_eff != K_req:
               print(f"[info] Reducing basis_K from {K_req} to {K_eff} (≤ L={L}).")
           B = Vt[:K_eff, :]  # (K_eff, L)
        else:
           B = None
        return (None if B is None else B.detach().cpu(),
                mu_log.squeeze(0).detach().cpu(),
                sd_log.squeeze(0).detach().cpu())

    # --------------- run ---------------
    def run(self, dryrun=False, basis_K: int | None = None):
        _seed_everything(self.split_seed)

        # datasets
        print('... Building datasets with a global seeded split')
        ds_train_flat, ds_valid_flat, split = self._build_datasets_global_split()
        train_ds = XYFromFlat(ds_train_flat)
        valid_ds = XYFromFlat(ds_valid_flat)

        # define some dimensions
        L = len(self.label_o)
        d_phys = (len(self.label_i) - 2) if self.inputs_have_avrv else len(self.label_i)
        d_full = len(self.label_i)

        # Quick distribution sanity check (inputs only)
        def _summ(ds, d_full, cap=20000):
            X = []
            n = min(len(ds), cap)
            for i in range(n):
                X.append(ds[i][:d_full].cpu().numpy())
            X = np.vstack(X)
            return (X.mean(0), X.std(0), X.min(0), X.max(0))

        mT, sT, loT, hiT = _summ(ds_train_flat, d_full)
        mV, sV, loV, hiV = _summ(ds_valid_flat, d_full)

        print("---- Input distribution check (first few shown) ----")
        print("train mean/std:", np.round(mT, 6), np.round(sT, 6))
        print("train range   :", np.round(loT, 6), np.round(hiT, 6))
        print("valid mean/std:", np.round(mV, 6), np.round(sV, 6))
        print("valid range   :", np.round(loV, 6), np.round(hiV, 6))
        print("----------------------------------------------------")

        # DataLoaders
        print('... Building DataLoaders')
        n_train, n_valid = len(train_ds), len(valid_ds)
        bs_train = min(self.batchsize, max(1, n_train))
        bs_valid = min(self.batchsize, max(1, n_valid))
        
        n_train, n_valid = len(train_ds), len(valid_ds)
        print(f"[dbg] n_train={n_train}, n_valid={n_valid}, bs_train={bs_train}, bs_valid={bs_valid}")
        
        train_loader = DataLoader(train_ds,
                                  sampler=RandomSampler(train_ds),
                                  batch_size=bs_train, num_workers=self.num_workers,
                                  pin_memory=(device.type=='cuda'), drop_last=False)
        valid_loader = DataLoader(valid_ds,
                                  sampler=SequentialSampler(valid_ds),
                                  batch_size=bs_valid, num_workers=self.num_workers,
                                  pin_memory=(device.type=='cuda'), drop_last=False)


        # optional PCA basis (in log-flux space) + mean log-spectrum
        basis_B, mu_log, sd_log = self._maybe_build_basis(ds_train_flat, K=basis_K)

        # model
        model = SpectralMLP_v1(
            d_phys=d_phys,
            d_full=d_full,
            L=L,
            H1=self.H1, H2=self.H2, H3=self.H3,
            W_k=self.W_k, W_resid=self.W_resid,
            basis_B=basis_B,
            mu_log=mu_log,
            include_extinction=self.include_extinction,
            include_resid=self.include_resid,
            inputs_have_avrv=self.inputs_have_avrv,
        ).to(device)

        # build per-pixel weights in log space (unit-mean) if requested
        # w_λ = 1 / (σ_log(λ)^2 + eps); normalize to mean 1 so the scalar loss scale is unchanged
        if self.weight_by_logvar and (sd_log is not None):
            sd = torch.as_tensor(sd_log, dtype=torch.float32, device=device).view(-1)  # (L,)
            w_pix = 1.0 / (sd * sd + self.weight_eps)
            # w_pix = (sd**2)
            w_pix = w_pix / w_pix.mean()
        else:
            w_pix = None  # no weighting

        # (optional) restart
        start_epoch = 0
        best_val = float('inf')

        if self.restartfile and os.path.isfile(self.restartfile):
            print(f"... Restarting from {self.restartfile}")
            load_state_dict_from_h5(model, self.restartfile,
                                    group="model", strict=True, dtype=torch.float32)

            if self.resume_from_restart:
                # try to read previous training metadata
                try:
                    with h5py.File(self.restartfile, "r") as h5:
                        if "meta" in h5:
                            g = h5["meta"]
                            prev_epochs = int(g.attrs.get("epochs_trained", 0))
                            prev_best   = float(g.attrs.get("best_valid", np.inf))
                            start_epoch = prev_epochs
                            best_val    = prev_best
                            print(f"... Resuming from epoch {start_epoch}, "
                                  f"previous best_valid={best_val:.4e}")
                        else:
                            print("... No 'meta' group found in restartfile; starting from epoch 0.")
                except Exception as e:
                    print(f"... Warning: failed to read meta from restartfile: {e}")
                    start_epoch = 0
                    best_val = float('inf')

        print('Model Arch:\n', model)
        total = _nparams(_unwrap(model))
        print(f"... Trainable parameters: {total} [{_fmt(total)}]")
                
        # optimizer + warmup → cosine decay schedule
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            (no_decay if (n.endswith("bias") or "ln" in n or "norm" in n) else decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 1e-4},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.lr, betas=(0.9, 0.999), fused=(device.type=='cuda')
        )
        total_epochs = self.numepochs
        warmup = int(self.warmup_epochs)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0/max(1, warmup), end_factor=1.0, total_iters=max(1, warmup))
        cos   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_epochs - warmup), eta_min=self.eta_min)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[max(1, warmup)])

        # mixed precision
        scaler = torch.amp.GradScaler(enabled=(device.type=='cuda'))
        def autocast_ctx(): return torch.amp.autocast(device_type="cuda", enabled=(device.type=='cuda'))

        # loss in log10 flux
        huber = nn.SmoothL1Loss(reduction="mean")

        # save metadata (labels & wavelengths)
        with h5py.File(self.outfilename, 'a') as h5:
            for key in ('label_i','label_o'):
                if key in h5: del h5[key]
            h5.create_dataset('label_i', data=np.array([s.encode('ascii') for s in self.label_i]))
            h5.create_dataset('label_o', data=np.array([s.encode('ascii') for s in self.label_o]))
            gmeta = h5.require_group('meta')
            gmeta.attrs['created']   = str(datetime.now())
            gmeta.attrs['trainper']  = float(self.trainper)
            gmeta.attrs['batchsize'] = int(self.batchsize)
            gmeta.attrs['lr']        = float(self.lr)
            gmeta.attrs['norm']      = bool(self.norm_outputs)
            gmeta.attrs['modpath']   = str(self.modpath)
            # store wavelength grid (Angstrom)
            if 'wavelengths_A' in h5: del h5['wavelengths_A']
            h5.create_dataset('wavelengths_A', data=np.asarray(ds_train_flat.wavelengths_A, dtype=np.float64))

        # ---- logging setup ----
        fig_loss, ax_loss = plt.subplots(nrows=3, ncols=1, figsize=(7,10), layout='constrained')
        for ax in ax_loss:
            ax.set_xlim(0, self.numepochs)
        ax_loss[0].set_ylabel('log(loss mean)')
        ax_loss[1].set_ylabel('log(loss std)')
        ax_loss[2].set_ylabel('log(loss median)')
        ax_loss[2].set_xlabel('Epoch')

        batchloss_arr, batchloss_std, batchloss_med = [], [], []
        validloss_arr, validloss_std, validloss_med = [], [], []

        def _safe_log10(x):
            import numpy as _np
            return _np.log10(_np.maximum(_np.asarray(x, float), 1e-12))


        # training loop
        coeff_w = 0.05  # auxiliary coefficient-alignment loss weight
        print('----- Starting Spectral Training Loop -----')
        for epoch in range(self.numepochs):
            t0 = time.time()
            model.train()
            losses = []

            # optional F0 freeze for first 50 epochs — ONLY if some other lane can learn
            # (i.e., extinction and/or residual lanes enabled). In intrinsic-only training
            # both lanes are off, so keep f0 trainable to preserve gradients.
            warmup_freeze = (epoch < 50)
            has_other_trainable_lane = (self.include_extinction or self.include_resid)
            if warmup_freeze and has_other_trainable_lane:
                model.freeze_f0(True)
            else:
                model.freeze_f0(False)

            opt.zero_grad(set_to_none=True)
            for step, (xb, yb) in enumerate(train_loader, start=1):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # split x,y
                xin = xb[:, :d_full]
                y   = yb  # flux (since ReadSpec norm=False)

                # log10 targets
                ylog = torch.log10(torch.clamp(y, min=1e-12))

                with autocast_ctx():
                    yhat_log = model(xin)                        # (B, L)

                    # ---- 1) log-space SmoothL1 (as before) ----
                    if w_pix is None:
                        log_huber = huber(yhat_log, ylog)        # scalar
                    else:
                        per_el_log = F.smooth_l1_loss(yhat_log, ylog, reduction='none')  # (B, L)
                        log_huber  = (per_el_log * w_pix).mean()

                    # ---- 2) fractional-error term computed in log space ----
                    if self.use_frac_loss:
                        # Δlog10 flux
                        delta_log = yhat_log - ylog                     # (B, L)
                        # clip to avoid insane ratios if predictions blow up
                        delta_log = torch.clamp(delta_log,
                                                min=-self.frac_clip_dex,
                                                max=self.frac_clip_dex)
                        # ratio = f_pred / f_true
                        ratio = torch.pow(10.0, delta_log)             # (B, L)
                        frac  = ratio - 1.0                            # (B, L)

                        # Option: use MSE on fractional error; could also use SmoothL1 if you prefer
                        if w_pix is None:
                            frac_mse = torch.mean(frac * frac)
                        else:
                            frac_mse = torch.mean((frac * frac) * w_pix)

                        loss_data = ((1.0 - self.frac_alpha) * log_huber +
                                    self.frac_alpha * frac_mse)
                    else:
                        loss_data = log_huber

                    # ---- 3) coefficient-alignment term (unchanged) ----
                    coeff_loss = 0.0
                    if hasattr(model, "f0") and hasattr(model.f0, "B"):   # low-rank head present
                        mu = model.mu_log if (hasattr(model, "mu_log") and isinstance(model.mu_log, torch.Tensor)) else 0.0
                        Bt = model.f0.B.B.t()                              # (L, K)

                        z_true = (ylog - mu) @ Bt                          # (B, K)
                        z_hat  = (yhat_log - mu) @ Bt                      # (B, K)

                        coeff_loss = F.mse_loss(z_hat, z_true, reduction='mean')

                    loss = (loss_data + coeff_w * coeff_loss) / max(1, self.grad_accum_steps)


                # accumulate grads
                scaler.scale(loss).backward()

                if (step % self.grad_accum_steps) == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                losses.append(loss.detach().item())
                
            train_mean = float(np.mean(losses))
            train_std  = float(np.std(losses)) if len(losses) else 0.0
            train_med  = float(np.median(losses)) if len(losses) else train_mean

            batchloss_arr.append(train_mean)
            batchloss_std.append(train_std)
            batchloss_med.append(train_med)

            # if we ended mid-accumulation, take one final optimizer step
            if (step % self.grad_accum_steps) != 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            # validation (every epoch)
            model.eval()
            v_losses = []
            with torch.inference_mode(), autocast_ctx():
                for xb, yb in valid_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    ylog = torch.log10(torch.clamp(yb, min=1e-12))

                    yhat_log = model(xb[:, :d_full])

                    # 1) log-space term
                    if w_pix is None:
                        log_huber_v = huber(yhat_log, ylog)
                    else:
                        per_el_log_v = F.smooth_l1_loss(yhat_log, ylog, reduction='none')
                        log_huber_v  = (per_el_log_v * w_pix).mean()

                    # 2) fractional term in log space
                    if self.use_frac_loss:
                        delta_log_v = yhat_log - ylog
                        delta_log_v = torch.clamp(delta_log_v,
                                                min=-self.frac_clip_dex,
                                                max=self.frac_clip_dex)
                        ratio_v = torch.pow(10.0, delta_log_v)
                        frac_v  = ratio_v - 1.0

                        if w_pix is None:
                            frac_mse_v = torch.mean(frac_v * frac_v)
                        else:
                            frac_mse_v = torch.mean((frac_v * frac_v) * w_pix)

                        v = ((1.0 - self.frac_alpha) * log_huber_v +
                            self.frac_alpha * frac_mse_v)
                    else:
                        v = log_huber_v

                    # coefficient alignment term (unchanged)
                    if hasattr(model, "f0") and hasattr(model.f0, "B"):
                        mu = model.mu_log if (hasattr(model, "mu_log") and isinstance(model.mu_log, torch.Tensor)) else 0.0
                        Bt = model.f0.B.B.t()
                        z_true = (ylog - mu) @ Bt
                        z_hat  = (yhat_log - mu) @ Bt
                        coeff_loss_val = F.mse_loss(z_hat, z_true, reduction='mean')
                        v = v + coeff_w * coeff_loss_val

                    v_losses.append(v.detach().item())

            val_mean = float(np.mean(v_losses)) if v_losses else float('inf')
            val_std  = float(np.std(v_losses)) if v_losses else 0.0
            val_med  = float(np.median(v_losses)) if v_losses else val_mean

            validloss_arr.append(val_mean)
            validloss_std.append(val_std)
            validloss_med.append(val_med)

            # checkpoint on best val
            if val_mean < best_val:
                best_val = val_mean
                save_state_dict_to_h5(_unwrap(model).state_dict(), self.outfilename, group="model", compression="gzip")
                save_meta_to_h5(self.outfilename,
                                n_inputs=d_full, n_outputs=L, nn_type="SpectralMLP_v1",
                                best_valid=float(best_val), epochs_trained=int(epoch+1),
                                date=str(datetime.now()))
                # convenience duplicate for non-PyTorch consumers (optional)
                try:
                    with h5py.File(self.outfilename, "a") as h5:
                        if "mu_log" in h5:
                            del h5["mu_log"]
                        m = _unwrap(model).mu_log
                        if m is not None:
                            h5.create_dataset("mu_log", data=m.detach().cpu().numpy(), compression="gzip")
                except Exception:
                    pass

            sched.step()

            if (epoch % self.plotevery == 0) or (epoch == self.numepochs - 1):
                bx = np.arange(len(batchloss_arr))
                vx = np.arange(len(validloss_arr))
                # mean
                ax_loss[0].plot(bx, _safe_log10(batchloss_arr), c='C0', lw=0.8, label='train' if epoch==0 else None)
                ax_loss[0].plot(vx, _safe_log10(validloss_arr), c='C3', lw=0.8, label='valid' if epoch==0 else None)
                # std
                ax_loss[1].plot(bx, _safe_log10(batchloss_std), c='C0', lw=0.8)
                ax_loss[1].plot(vx, _safe_log10(validloss_std), c='C3', lw=0.8)
                # median
                ax_loss[2].plot(bx, _safe_log10(batchloss_med), c='C0', lw=0.8)
                ax_loss[2].plot(vx, _safe_log10(validloss_med), c='C3', lw=0.8)
                if epoch == 0:
                    ax_loss[0].legend(loc='best', fontsize=9)
                for ii,ax in enumerate(ax_loss):
                    ax.set_xlim(0, epoch + 1)

                    if ii == 0:
                        x1 = np.array(_safe_log10(batchloss_arr))[np.isfinite(batchloss_arr)]
                        x2 = np.array(_safe_log10(validloss_arr))[np.isfinite(validloss_arr)]
                        minval = np.percentile(np.concatenate([x1, x2]), 5)
                        maxval = np.percentile(np.concatenate([x1, x2]), 95)
                    if ii == 1:
                        x1 = np.array(_safe_log10(batchloss_std))[np.isfinite(batchloss_std)]
                        x2 = np.array(_safe_log10(validloss_std))[np.isfinite(validloss_std)]
                        minval = np.percentile(np.concatenate([x1, x2]), 5)
                        maxval = np.percentile(np.concatenate([x1, x2]), 95)
                    if ii == 2:
                        x1 = np.array(_safe_log10(batchloss_med))[np.isfinite(batchloss_med)]
                        x2 = np.array(_safe_log10(validloss_med))[np.isfinite(validloss_med)]
                        minval = np.percentile(np.concatenate([x1, x2]), 5)
                        maxval = np.percentile(np.concatenate([x1, x2]), 95)

                    ax.set_ylim(minval - 0.1 * (maxval - minval), maxval + 0.1 * (maxval - minval))

                out_png = f'{self.plotdir}/{os.path.split(self.outfilename)[-1].replace(".h5","")}_loss.png'
                fig_loss.savefig(out_png, dpi=150)

            # print epoch summary every 25 epochs
            if epoch % 25 == 0 or epoch == self.numepochs - 1:
                print(f"... Epoch {epoch+1}/{self.numepochs}  "
                    f"train_logHuber={math.log10(max(train_mean,1e-12)):.5f}  "
                    f"valid_logHuber={math.log10(max(val_mean,1e-12)):.5f}  "
                    f"loss={loss_data.item():.4e}  "
                     f"zloss={float(coeff_loss) if not isinstance(coeff_loss, torch.Tensor) else float(coeff_loss.detach().cpu()) :.4e}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  "
                    f"time={time.time()-t0:.1f}s")

            if self.early_stopping and (val_mean >= best_val - self.early_stopping_min_delta):
                # rudimentary patience: stop if no new best for N epochs
                # (simple version; add a counter if you want exact behavior as photometric)
                pass

        plt.close(fig_loss)

        torch.cuda.empty_cache()
        print('Finished spectral training.')
        return model