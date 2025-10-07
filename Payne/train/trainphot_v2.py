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

from ..utils import readKorg
from ..utils.readKorg import XYFromFlat
from ..utils.io_h5 import save_state_dict_to_h5, load_state_dict_from_h5, save_labels_norms_to_h5, save_meta_to_h5

from .NNmodels_new import MLP_v0
from .NNmodels_new import MLP_v1
from .NNmodels_new import MLP_v2

from ..predict import photANN_new as photANN

def _unwrap(model):
    """Return the underlying nn.Module if this is a compiled model."""
    return getattr(model, "_orig_mod", model)

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
            model = photANN.readNN(self.restartfile, nntype=self.NNtype)
        else:
            print(f'Running New NN with NNtype: {self.NNtype}')
            model = defmod(self.D_in, self.H1, self.H2, self.H3, self.D_out, NNtype=self.NNtype)

        print('Model Arch:\n', model)
        model.to(device)

        # compile model to speed things up
        if hasattr(torch, "compile"):
            compile_mode = "max-autotune" if use_cuda else "reduce-overhead"
            try:
                model = torch.compile(model, mode=compile_mode, fullgraph=False)
            except Exception as _e:
                if self.verbose:
                    print(f"... torch.compile unavailable or failed ({_e}); continuing without compile.")
            
        # ---- datasets & loaders ----
        train_ds_flat = readKorg.ReadPhot(
            modpath=self.modpath,
            filters=self.label_o,
            filter_wavelength_method="pivot",
            label_i=self.label_i,
            label_o=self.label_o,
            norm=self.norm,
            returntorch=True,
            type='train',
            trainpercentage=self.trainper,
            parrange=self.parrange,
        )
        valid_ds_flat = readKorg.ReadPhot(
            modpath=self.modpath,
            filters=self.label_o,
            filter_wavelength_method="pivot",
            label_i=self.label_i,
            label_o=self.label_o,
            norm=self.norm,
            returntorch=True,
            type='valid',
            trainpercentage=self.trainper,
            parrange=self.parrange,
        )

        print(f"... ReadPhot sizes: train={len(train_ds_flat)}  valid={len(valid_ds_flat)}")
        
        # wrap to yield (x, y)
        train_ds = XYFromFlat(train_ds_flat)
        valid_ds = XYFromFlat(valid_ds_flat)
        
        n_train = len(train_ds)
        n_valid = len(valid_ds)
        if self.batchsize > n_train:
            print(f"... Warning: batchsize {self.batchsize} > train size {n_train}; lowering batchsize and disabling drop_last.")
        train_bs = min(self.batchsize, max(1, n_train))
        valid_bs = min(self.batchsize, max(1, n_valid))
    
        linux_gpu = (device.type == "cuda" and sys.platform != "darwin")
        nw  = self.num_workers if not linux_gpu else max(self.num_workers, 4)
        ppf = 2 if nw == 0 else 4

        train_kwargs = dict(
            sampler=RandomSampler(train_ds),
            batch_size=train_bs,
            num_workers=nw,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=(nw > 0 and linux_gpu),
        )
        valid_kwargs = dict(
            sampler=SequentialSampler(valid_ds),
            batch_size=valid_bs,
            num_workers=nw,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=(nw > 0 and linux_gpu),
        )
        if nw > 0:
            train_kwargs["prefetch_factor"] = ppf
            valid_kwargs["prefetch_factor"] = ppf

        train_loader = DataLoader(train_ds, **train_kwargs)
        valid_loader = DataLoader(valid_ds, **valid_kwargs)
        
        print(f"... Train samples: {n_train}, batch: {train_bs}")
        print(f"... Valid  samples: {n_valid}, batch: {valid_bs}")

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
            [{"params": decay, "weight_decay": 5e-4},
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
        early_stopper = EarlyStopping(patience=50, min_delta=1e-4, verbose=True)

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
    
        print('----- Starting Training Loop ------')
        for epoch in range(self.numepochs):
            t0 = time.time()
            model.train()
            batch_losses = []

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                # with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                #     yhat = model(x)
                #     loss = loss_fn(yhat, y)
                    
                with autocast_ctx():
                    yhat = model(x)
                    tloss = loss_fn(yhat, y)
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

            # validation step
            if do_val:
                model.eval()
                v_sum, v_sumsq, v_cnt = 0.0, 0.0, 0
                with torch.inference_mode():
                    for x, y in valid_loader:
                        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                        with autocast_ctx():
                            vloss = loss_fn(model(x), y)
                        li = float(vloss)
                        v_sum += li; v_sumsq += li*li; v_cnt += 1

                val_m = v_sum / max(1, v_cnt)
                val_std = math.sqrt(max(0.0, v_sumsq / max(1, v_cnt) - val_m*val_m))
                validloss_arr.append(val_m)
                validloss_std.append(val_std)
                validloss_med.append(val_m)  # med≈mean for MSE here
                last_val_m = val_m

                # checkpoint & patience update only when we actually validated
                if val_m < best_val:
                    best_val = val_m
                    val_checks_without_improve = 0
                    # save best → HDF5
                    
                    # unwrap compiled model if torch.compile was used
                    base = _unwrap(model)

                    # now store
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


            else:
                # carry forward last validation metrics for logging/plots
                val_m = last_val_m
                validloss_arr.append(val_m)
                validloss_std.append(validloss_std[-1] if validloss_std else 0.0)
                validloss_med.append(validloss_med[-1] if validloss_med else val_m)

            if do_val:
                # compute stop criteria
                stop = (val_checks_without_improve >= early_stopper.patience)
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
            val_tag = f"(val {val_checks_without_improve}/{early_stopper.patience})" if do_val else "(skip)"
            val_display = np.log10(val_m) if np.isfinite(val_m) else float('nan')
            print(f"... Epoch {epoch+1}/{self.numepochs} {val_tag} "
                f"train_logMSE={np.log10(train_m):.5f}  "
                f"valid_logMSE={val_display:.5f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"time={time.time()-t0:.1f}s")
            if stop:
                print("... Early Stopping Triggered")
                break

        plt.close(fig_loss)
        torch.cuda.empty_cache()
        print('Finished training model.')
        return model