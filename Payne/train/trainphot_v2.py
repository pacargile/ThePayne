import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if str(device) != "cpu":
    dtype = torch.cuda.FloatTensor
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps:0")
    dtype = torch.FloatTensor

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Reserved: ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print()

from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,ExponentialLR
from torch.utils.data import DataLoader,SubsetRandomSampler, RandomSampler, SequentialSampler

import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Pool

from astropy.table import Table,vstack

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import traceback
import numpy as np
from scipy.stats import scoreatpercentile
import warnings
import h5py
import time,sys,os,glob,shutil
from datetime import datetime
import random
import math
import os, sys

from ..utils import readKorg_old
from ..utils.readKorg_old import XYFromFlat
from ..utils.io_h5 import save_state_dict_to_h5, load_state_dict_from_h5, save_labels_norms_to_h5, save_meta_to_h5

from .NNmodels_new import MLP_v0
from .NNmodels_new import MLP_v1
from .NNmodels_new import MLP_v2

from ..predict import photANN_new as photANN

def _unwrap(model):
    """Return the underlying nn.Module if this is a compiled model."""
    return getattr(model, "_orig_mod", model)

def _nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def _fmt(n):
    if n >= 1e6:  return f"{n/1e6:.2f} M"
    if n >= 1e3:  return f"{n/1e3:.2f} K"
    return str(n)

def should_validate(epoch, total_epochs, train_loss_hist):
    """
    Decide whether to run validation this epoch.
    - Every 5 epochs in the first half
    - Every 2 epochs in the middle
    - Every epoch in the final 15% of training
    - Also force validation if train loss improved 'a lot' since last check
    """
    e = epoch + 1
    # final phase: dense validation
    if e > int(0.85 * total_epochs):
        return True
    # regular cadence
    if e <= total_epochs // 2:
        cadence = 5
    else:
        cadence = 2

    if e % cadence == 0:
        return True

    # adaptive trigger: if last 3 train losses improved > thresh, check early
    if len(train_loss_hist) >= 4:
        recent = train_loss_hist[-4:]
        if recent[-1] < min(recent[:-1]) - 1e-3:  # tune threshold
            return True

    return False

class EarlyStopping:
    def __init__(self, patience=100, min_delta=0.0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): Print when early stopping is triggered.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, current_loss):
        if (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"... EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

def defmod(D_in,H1,H2,H3,D_out,NNtype='MLP_v0'):
    if NNtype == 'MLP_v0':
        return MLP_v0(D_in,H1,H2,H3,D_out)
    elif NNtype == 'MLP_v1':
        return MLP_v1(D_in,H1,H2,H3,D_out)
    elif NNtype == 'MLP_v2':
        return MLP_v2(D_in,H1,H2,H3,D_out)
    else:
        raise ValueError(f"Unknown NNtype: {NNtype}")

def _seed_everything(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class TrainMod(object):
    """docstring for TrainMod"""
    def __init__(self, *arg, **kwargs):
        super(TrainMod, self).__init__()

        print(f'... Start Training Code at {datetime.now()}')
        sys.stdout.flush()

        # --- reproducible data split seed ---
        self.split_seed = kwargs.get('split_seed', 1337)

        # ---- logging / plotting
        self.logplot   = kwargs.get('logplot', True)

        # ---- training config
        self.trainper  = kwargs.get('trainper', 0.9)
        self.numepochs = kwargs.get('numepochs', 10000)
        self.batchsize = kwargs.get('batchsize', 2048)
        self.lr        = kwargs.get('lr', 1e-3)
        self.NNtype    = kwargs.get('NNtype', 'MLP_v0')
        self.plotevery = kwargs.get('plotevery', 20)
        self.plotdir   = kwargs.get('plotdir', './plots/')

        # --- two-phase validation knobs ---
        self.stress_every      = kwargs.get("stress_every", 5)  # run stress test every N validations
        self.stress_avgrid     = kwargs.get("stress_avgrid",
                                            [0.0, 0.5, 1, 2, 3, 5, 10, 20, 50, 75, 100])
        self.stress_rvgrid     = kwargs.get("stress_rvgrid", [2.5, 3.1, 4.0, 5.0])
        self.stress_limit      = kwargs.get("stress_limit", 20000)   # cap number of stress samples
        self.stress_max_batches= kwargs.get("stress_max_batches", 200)  # or None
        self.stress_seed       = kwargs.get("stress_seed", 1234)     # reproducible subsetting

        # ---- Early Stopping Parameters
        self.early_stopping = kwargs.get('early_stopping', True)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 100)
        self.early_stopping_min_delta = kwargs.get('early_stopping_min_delta', 1e-5)

        print(f'... Early Stopping: {self.early_stopping}, {self.early_stopping_patience}, {self.early_stopping_min_delta}')

        # if verbose
        self.verbose = kwargs.get('verbose', True)

        # create plot directory if it doesn't exist
        os.makedirs(self.plotdir, exist_ok=True)

        # network widths
        self.H1 = kwargs.get('H1', 256)
        self.H2 = kwargs.get('H2', 256)
        self.H3 = kwargs.get('H3', 256)

        # ---- labels (defaults aligned to new grids)
        self.label_i = kwargs.get('label_i', ['logt','logg','feh','afe','av','rv'])
        # by default, expect caller to pass label_o; if not, set a safe placeholder or raise
        self.label_o = kwargs.get('label_o', None)
        if self.label_o is None or len(self.label_o) == 0:
            raise ValueError("Please provide label_o (e.g., ['gaia_g','gaia_bp','gaia_rp']).")

        self.D_in  = len(self.label_i)
        self.D_out = len(self.label_o)

        # ---- parameter ranges (keys should match label_i fields)
        # These are *only* used by ReadPhot when provided via parrange
        self.parrange = kwargs.get('parrange', {
            'logt': [3.0, 4.7],     # ~1e3 K to 5e4 K; adjust as needed
            'logg': [-2.0, 6.0],
            'feh':  [-5.0, 1.0],
            'afe':  [-0.2, 0.6],
            'av':   [0.0, 20.0],
            'rv':   [2.0, 6.0],
        })

        # ---- files & flags
        self.restartfile = kwargs.get('restartfile', None)
        if self.restartfile is not None:
            print(f'... Restarting File: {self.restartfile}')

        self.outfilename = kwargs.get('output', 'TRAIN_OUT.h5')
        self.modpath     = kwargs.get('modpath', './cwc_models.h5')

        self.num_workers = kwargs.get('num_workers', 0)  # 0 avoids MP headaches; bump later on Linux

        # normalization flag (ReadPhot applies/records means/stds)
        self.norm = kwargs.get('norm', True)
        print(f'... Running with normalized labels: {self.norm}')

        print('... Running Training on Device: {}'.format(device))

        # ---- initialize (or update) the output HDF5 with labels + basic meta only
        # norms and model weights are saved later during training when validation improves
        try:
            with h5py.File(self.outfilename, 'a') as outfile_i:
                # labels
                if 'label_i' in outfile_i: del outfile_i['label_i']
                if 'label_o' in outfile_i: del outfile_i['label_o']
                outfile_i.create_dataset('label_i',
                    data=np.array([x.encode("ascii","ignore") for x in self.label_i]))
                outfile_i.create_dataset('label_o',
                    data=np.array([x.encode("ascii","ignore") for x in self.label_o]))

                # meta attrs (lightweight)
                meta = outfile_i.get('meta', None) or outfile_i.create_group('meta')
                meta.attrs['created']   = str(datetime.now())
                meta.attrs['nn_type']   = str(self.NNtype)
                meta.attrs['trainper']  = float(self.trainper)
                meta.attrs['batchsize'] = int(self.batchsize)
                meta.attrs['lr']        = float(self.lr)
                meta.attrs['norm']      = bool(self.norm)
                meta.attrs['modpath']   = str(self.modpath)
        except Exception:
            print('!!! PROBLEM INITIALIZING OUTPUT HDF5 !!!')
            raise

        print(f'... Din: {self.D_in}, Dout: {self.D_out}')
        print(f'... Input Labels:  {self.label_i}')
        print(f'... Output Labels: {self.label_o}')
        print('... Finished Init')
        sys.stdout.flush()
        
    def __call__(self, dryrun=False):
        '''
        call instance so that train_pixel can be called with multiprocessing
        and still have all of the class instance variables

        '''
        try:
            return self.train_mod(dryrun=dryrun)
        except Exception as e:
            traceback.print_exc()
            print()
            raise e

    def run(self, dryrun=False):
        '''
        function to actually run the training on models

        dryrun: bool
            if True, then just return the model, don't train

        '''
        # start total timer
        tottimestart = datetime.now()

        print('Starting Training at {0}'.format(tottimestart))
        sys.stdout.flush()

        net = self(dryrun=dryrun)

        tottimeend = datetime.now()

        print('Finished Training at {0} ({1})'.format(tottimeend,tottimeend-tottimestart))
        return net


    def train_mod(self, dryrun=False):
        _seed_everything(1337)

        # determine if cuda is available
        use_cuda = (device.type == "cuda")

        if use_cuda:
            print(f'Running on GPU {torch.cuda.current_device()+1}/{torch.cuda.device_count()}')

        # ---- model (new or restart) ----
        if self.restartfile is not None and os.path.isfile(self.restartfile):
            print(f'Restarting from: {self.restartfile} ({self.NNtype})')
            # Construct the architecture from your *current* config
            model = defmod(self.D_in, self.H1, self.H2, self.H3, self.D_out, NNtype=self.NNtype)
            # Load weights from HDF5 (strict)
            load_state_dict_from_h5(model, self.restartfile, group="model", strict=True, dtype=torch.float32)
        else:
            print(f'Running New NN with NNtype: {self.NNtype}')
            model = defmod(self.D_in, self.H1, self.H2, self.H3, self.D_out, NNtype=self.NNtype)
            
        print('Model Arch:\n', model)
        model.to(device)

        # # compile model to speed things up
        # if hasattr(torch, "compile"):
        #     compile_mode = "max-autotune" if use_cuda else "reduce-overhead"
        #     try:
        #         model = torch.compile(model, mode=compile_mode, fullgraph=False)
        #     except Exception as _e:
        #         if self.verbose:
        #             print(f"... torch.compile unavailable or failed ({_e}); continuing without compile.")

        try:
            base = _unwrap(model)  # you already use this when saving
        except NameError:
            base = model

        total = _nparams(base)
        print(f"... Trainable parameters (total): {total}  [{_fmt(total)}]")

        # per-head breakdown if present
        if hasattr(base, "f0"):
            n = _nparams(base.f0)
            print(f"    ├─ f0 (stellar head): {n}  [{_fmt(n)}]")
        if hasattr(base, "khat"):
            n = _nparams(base.khat)
            print(f"    ├─ khat (extinction lane): {n}  [{_fmt(n)}]")
        if hasattr(base, "resid"):
            n = _nparams(base.resid)
            print(f"    └─ resid (small residual): {n}  [{_fmt(n)}]")

        # ---- datasets & loaders ----
        # Build ONE anchor dataset to define the split & training normalization
        anchor_train_ds = readKorg_old.ReadPhot(
            modpath=self.modpath,
            filters=self.label_o,
            filter_wavelength_method="pivot",
            label_i=self.label_i,
            label_o=self.label_o,
            norm=self.norm,                 # compute training norms here
            returntorch=True,
            type='train',
            trainpercentage=self.trainper,
            parrange=self.parrange,
            extinction_mode="sample",
            split_seed=self.split_seed,     # deterministic split
        )

        # Extract split indices and the training normalization
        split = anchor_train_ds.split_indices            # {'train','valid','test'} of model_index values
        train_norms = dict(anchor_train_ds.normfactor)   # {label: (mean, std)}

        # Reuse the anchor as the training dataset
        train_ds_flat = anchor_train_ds

        # Build VALID with identical rows and identical normalization
        valid_ds_flat = readKorg_old.ReadPhot(
            modpath=self.modpath,
            filters=self.label_o,
            filter_wavelength_method="pivot",
            label_i=self.label_i,
            label_o=self.label_o,
            norm=self.norm,
            normfactor=train_norms,          # force training norms
            returntorch=True,
            type='valid',
            trainpercentage=self.trainper,   # ignored once split=... is given; kept for clarity
            parrange=self.parrange,
            extinction_mode="fixed",
            fixed_av=0.0,
            fixed_rv=3.1,
            split_seed=self.split_seed,      
            split=split,                     # force same rows as anchor
        )

        print(f"... ReadPhot sizes: train={len(train_ds_flat)}  valid={len(valid_ds_flat)}")

        # Wrap to (x,y)
        train_ds = XYFromFlat(train_ds_flat)
        valid_ds = XYFromFlat(valid_ds_flat)
        
        n_train = len(train_ds)
        n_valid = len(valid_ds)
        if self.batchsize > n_train:
            print(f"... Warning: batchsize {self.batchsize} > train size {n_train}; lowering batchsize and disabling drop_last.")
        train_bs = min(self.batchsize, max(1, n_train))
        valid_bs = min(self.batchsize, max(1, n_valid))
    
        # linux_gpu = (device.type == "cuda" and sys.platform != "darwin")
        nw  = 0 # self.num_workers if not linux_gpu else max(self.num_workers, 4)
        # ppf = 2 if nw == 0 else 4

        train_kwargs = dict(
            sampler=RandomSampler(train_ds),
            batch_size=train_bs,
            num_workers=nw,
            pin_memory=False,#(device.type == "cuda"),
            drop_last=False,
            persistent_workers=False,#(nw > 0 and linux_gpu),
        )
        valid_kwargs = dict(
            sampler=SequentialSampler(valid_ds),
            batch_size=valid_bs,
            num_workers=nw,
            pin_memory=False,#(device.type == "cuda"),
            drop_last=False,
            persistent_workers=False,#(nw > 0 and linux_gpu),
        )
        if nw > 0:
            train_kwargs["prefetch_factor"] = ppf
            valid_kwargs["prefetch_factor"] = ppf

        train_loader = DataLoader(train_ds, **train_kwargs)
        valid_loader = DataLoader(valid_ds, **valid_kwargs)
        
        print(f"... Train samples: {n_train}, batch: {train_bs}")
        print(f"... Valid  samples: {n_valid}, batch: {valid_bs}")

        # --- persist split & norms once per run (safe to overwrite) ---
        with h5py.File(self.outfilename, "a") as h5:
            gsplit = h5.require_group("split")
            for name in ("train_idx", "valid_idx", "test_idx"):
                if name in gsplit: del gsplit[name]
            gsplit.create_dataset("train_idx", data=split["train"], compression="gzip")
            gsplit.create_dataset("valid_idx", data=split["valid"], compression="gzip")
            gsplit.create_dataset("test_idx",  data=split["test"],  compression="gzip")
            gsplit.attrs["split_seed"] = int(self.split_seed)

            gnorms = h5.require_group("norms")
            # clear and write training norms (label_i + label_o)
            for k in list(gnorms.keys()):
                del gnorms[k]
            for k, (mu, sd) in train_norms.items():
                g = gnorms.require_group(k)
                g.attrs["mean"] = float(mu)
                g.attrs["std"]  = float(sd)

        # --- ensure the output file also contains the current weights ---
        # At this point, `model` already has weights loaded from `self.restartfile`
        # (since you swapped to defmod(...) + load_state_dict_from_h5(...)).
        # Save them into `self.outfilename` so eval can load from there.
        base = _unwrap(model)
        save_state_dict_to_h5(base.state_dict(), self.outfilename, group="model", compression="gzip")
        
        # --- lazy stress-test loader (built on first use and cached) ---
        _stress_loader = None
        def get_stress_loader():
            nonlocal _stress_loader
            if _stress_loader is not None:
                return _stress_loader

            stress_ds_flat = readKorg_old.ReadPhot(
                modpath=self.modpath,
                filters=self.label_o,
                filter_wavelength_method="pivot",
                label_i=self.label_i,
                label_o=self.label_o,
                norm=self.norm,
                normfactor=train_norms,          
                returntorch=True,
                type='valid',                    
                trainpercentage=self.trainper,
                parrange=self.parrange,
                extinction_mode="grid",
                avgrid=self.stress_avgrid,
                rvgrid=self.stress_rvgrid,
                split_seed=self.split_seed,
                split=split,
            )

            # cap size deterministically
            total = len(stress_ds_flat)
            rng = np.random.default_rng(self.stress_seed)
            if (self.stress_limit is not None) and (total > self.stress_limit):
                idx = rng.choice(total, size=self.stress_limit, replace=False)
                stress_ds = torch.utils.data.Subset(XYFromFlat(stress_ds_flat), idx)
            else:
                stress_ds = XYFromFlat(stress_ds_flat)

            # always single-process, non-pinned to avoid HDF5+fork issues
            stress_loader = DataLoader(
                stress_ds,
                batch_size=min(self.batchsize, 2048),
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
            )
            _stress_loader = stress_loader
            if self.verbose:
                print(f"... Built stress loader: base={total}, "
                    f"kept={len(stress_ds)} | Av×Rv = {len(self.stress_avgrid)}×{len(self.stress_rvgrid)}")
            return _stress_loader        


        # ---- loss & optimizer ----
        loss_fn = torch.nn.MSELoss(reduction='mean')

        # AdamW with decoupled weight decay; exclude norms & biases from decay
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if n.endswith("bias") or "ln" in n or "norm" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        # optimizer = torch.optim.AdamW(
        #     [{"params": decay, "weight_decay": 5e-4},
        #     {"params": no_decay, "weight_decay": 0.0}],
        #     lr=self.lr,
        #     betas=(0.9, 0.999)
        # )
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0}],
            lr=self.lr, betas=(0.9, 0.999),
            fused=(device.type == "cuda")
        )

        # cosine anneal with linear warmup (5% of epochs)
        total_epochs = self.numepochs
        warmup_epochs = max(1, int(0.05 * total_epochs))
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            # cosine from 1.0 → 0.1
            t = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        grad_clip = 1.0

        # ---- logging setup ----
        fig_loss, ax_loss = plt.subplots(nrows=3, ncols=1, figsize=(7,10), layout='constrained')
        for ax in ax_loss:
            ax.set_xlim(0, self.numepochs)
        ax_loss[0].set_ylabel('log(MSE)')
        ax_loss[1].set_ylabel('log(Std batch loss)')
        ax_loss[2].set_ylabel('log(Med batch loss)')
        ax_loss[2].set_xlabel('Epoch')

        if dryrun:
            return [model, optimizer, datetime.now() - datetime.now()]

        # ---- early stopping (persist across epochs) ----
        early_stopper = EarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta, verbose=True)

        batchloss_arr, batchloss_std, batchloss_med = [], [], []
        validloss_arr, validloss_std, validloss_med = [], [], []

        best_val = float("inf")

        # GradScaler (no device_type kwarg)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler(enabled=use_cuda)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)  # very old fallback

        # pick autocast
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            def autocast_ctx():
                # device_type goes here (not in GradScaler)
                return torch.amp.autocast(device_type="cuda", enabled=use_cuda)
        else:
            def autocast_ctx():
                return torch.cuda.amp.autocast(enabled=use_cuda)


        train_loss_hist = []
        last_val_m = float("inf")
        val_checks_without_improve = 0

        # define local function to compute validation metrics
        def run_validation(loader):
            model.eval()
            v_sum, v_sumsq, v_cnt = 0.0, 0.0, 0
            with torch.inference_mode():
                for b, (x, y) in enumerate(loader):
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                    with autocast_ctx():
                        vloss = loss_fn(model(x), y)
                    li = float(vloss)
                    v_sum += li; v_sumsq += li*li; v_cnt += 1
                    if loader is not valid_loader and self.stress_max_batches is not None and (b+1) >= self.stress_max_batches:
                        break
            val_m = v_sum / max(1, v_cnt)
            val_std = math.sqrt(max(0.0, v_sumsq / max(1, v_cnt) - val_m*val_m))
            return val_m, val_std, v_cnt
    
        print('----- Starting Training Loop ------')
        for epoch in range(self.numepochs):
            t0 = time.time()
            model.train()
            batch_losses = []

            # for x, y in train_loader:
            #     x = x.to(device, non_blocking=True)
            #     y = y.to(device, non_blocking=True)

            #     optimizer.zero_grad(set_to_none=True)
            #     # with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            #     #     yhat = model(x)
            #     #     loss = loss_fn(yhat, y)
                    
            #     with autocast_ctx():
            #         yhat = model(x)
            #         tloss = loss_fn(yhat, y)
            #     scaler.scale(tloss).backward()
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            #     scaler.step(optimizer)
            #     scaler.update()

            #     batch_losses.append(tloss.item())

            AV_IDX = 4   # x[:,4] is Av

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # we need autograd w.r.t. Av for the derivative lock
                Av = x[:, AV_IDX].clone().detach().to(device)
                Av.requires_grad_(True)
                x_mod = x.clone()
                x_mod[:, AV_IDX] = Av  # ensure this tensor is the one used in forward

                optimizer.zero_grad(set_to_none=True)

                with autocast_ctx():
                    # return k_hat so we don't re-run the model
                    yhat, k_hat = model(x_mod, return_khat=True)

                    # base photometry loss (Huber or MSE)
                    loss_data = loss_fn(yhat, y)

                    # -------- (1) Batchwise slope penalty: residual vs true magnitude --------
                    res = yhat - y                      # (B, D_out)
                    yt  = y - y.mean(dim=0, keepdim=True)
                    rt  = res - res.mean(dim=0, keepdim=True)
                    slope = (yt * rt).mean(dim=0) / (yt.pow(2).mean(dim=0) + 1e-8)
                    loss_slope = (slope.pow(2)).mean()  # scalar

                    # -------- (2) Av-linearity lock: ∂(m̂ − Av·k̂)/∂Av → 0 --------
                    # NOTE: This does not need y. It enforces that only k_hat carries Av.
                    m_minus_Avk = yhat - Av[:, None] * k_hat         # (B, D_out)
                    dAv = torch.autograd.grad(
                        m_minus_Avk.sum(), Av, create_graph=True, allow_unused=False
                    )[0]                                            # (B,)
                    loss_dAv = (dAv.pow(2)).mean()

                    # total loss (start with 1e-2; tune 3e-3–3e-2 if needed)
                    tloss = loss_data #+ 1e-2*loss_slope + 1e-2*loss_dAv

                scaler.scale(tloss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                batch_losses.append(tloss.item())

            # epoch stats
            train_m = float(np.mean(batch_losses))
            train_loss_hist.append(train_m)
            batchloss_arr.append(train_m)
            batchloss_std.append(float(np.std(batch_losses)))
            batchloss_med.append(float(np.median(batch_losses)))

            # validation decision
            do_val = should_validate(epoch, self.numepochs, train_loss_hist)

            if do_val:
                # --- primary, stable validation (fixed Av/Rv) ---
                val_m, val_std, val_batches = run_validation(valid_loader)
                validloss_arr.append(val_m)
                validloss_std.append(val_std)
                validloss_med.append(val_m)
                last_val_m = val_m

                # checkpoint & patience **based only on fixed validation**
                if val_m < best_val:
                    best_val = val_m
                    val_checks_without_improve = 0
                    base = _unwrap(model)
                    save_state_dict_to_h5(base.state_dict(), self.outfilename, group="model", compression="gzip")
                    save_labels_norms_to_h5(self.outfilename, self.label_i, self.label_o,
                                            normfactor=(train_ds_flat.normfactor if self.norm else None))
                    save_meta_to_h5(self.outfilename,
                                    n_inputs=len(self.label_i),
                                    n_outputs=len(self.label_o),
                                    nn_type=self.NNtype,
                                    best_valid_mse=float(val_m),
                                    epochs_trained=int(epoch + 1),
                                    date=str(datetime.now()))
                else:
                    val_checks_without_improve += 1

                # --- periodic stress test (small Av/Rv grid) ---
                do_stress = (self.stress_every is not None) and (self.stress_every > 0) and ((epoch % self.stress_every) == 0)
                if do_stress:
                    stress_loader = get_stress_loader()
                    stress_m, stress_std, stress_batches = run_validation(stress_loader)
                    # lightweight print; do not use for checkpoint decisions
                    print(f"    [stress] Av×Rv grid: mean MSE={stress_m:.6e} "
                        f"(std={stress_std:.6e}, batches={stress_batches})", flush=True)
            else:
                # carry forward last validation metrics for logging/plots
                val_m = last_val_m
                validloss_arr.append(val_m)
                validloss_std.append(validloss_std[-1] if validloss_std else 0.0)
                validloss_med.append(validloss_med[-1] if validloss_med else val_m)            
                        
            if do_val:
                if self.early_stopping:
                    # compute stop criteria
                    stop = (val_checks_without_improve >= early_stopper.patience)
                else:
                    stop = False
            else:
                stop = False

            scheduler.step()

            # plots (cheap)
            def _safe_log10(x): 
                return np.log10(np.maximum(np.asarray(x, float), 1e-12))

            if (epoch % self.plotevery == 0) or (epoch == self.numepochs - 1):
                bxarr = np.arange(len(batchloss_arr))
                vxarr = np.arange(len(validloss_arr))
                ax_loss[0].plot(bxarr, _safe_log10(batchloss_arr), c='C0', lw=0.8)
                ax_loss[0].plot(vxarr, _safe_log10(validloss_arr), c='C3', lw=0.8)
                ax_loss[1].plot(bxarr, _safe_log10(batchloss_std), c='C0', lw=0.8)
                ax_loss[1].plot(vxarr, _safe_log10(validloss_std), c='C3', lw=0.8)
                ax_loss[2].plot(bxarr, _safe_log10(batchloss_med), c='C0', lw=0.8)
                ax_loss[2].plot(vxarr, _safe_log10(validloss_med), c='C3', lw=0.8)
                if (epoch % 5 == 0) or (epoch == self.numepochs - 1):
                    outputplot = f'{self.plotdir}/{os.path.split(self.outfilename)[-1].replace(".h5","")}_loss.png'
                    fig_loss.savefig(outputplot, dpi=150)

            # early stop
            if self.early_stopping:
                val_tag = f"(val {val_checks_without_improve}/{early_stopper.patience})" if do_val else "(skip)"
            else:
                val_tag = "(No ES)"
            val_display = np.log10(val_m) if np.isfinite(val_m) else float('nan')
            if epoch % 25 == 0 or epoch == self.numepochs - 1:
                print(f"... Epoch {epoch+1}/{self.numepochs} {val_tag} "
                    f"train_logMSE={np.log10(train_m):.5f}  "
                    f"valid_logMSE={val_display:.5f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                    f"time={time.time()-t0:.1f}s")
            if stop:
                print("... Early Stopping Triggered")
                break

        # rescale x-axis in case of early stopping
        for ax in ax_loss:
            ax.set_xlim(0, epoch + 1)
        plt.close(fig_loss)
        torch.cuda.empty_cache()
        print('Finished training model.')
        return model