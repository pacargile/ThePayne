import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if str(device) != "cpu":
    dtype = torch.cuda.FloatTensor
    torch.backends.cudnn.benchmark = True
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
from torch.utils.data import DataLoader,SubsetRandomSampler

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

from ..utils import readKorg

from .NNmodels_new import MLP_v0
from .NNmodels_new import MLP_v1

from ..predict import photANN_new as photANN

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
        self.best_loss = np.inf
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

class TrainMod(object):
    """docstring for TrainMod"""
    def __init__(self, *arg, **kwargs):
        super(TrainMod, self).__init__()

        print(f'... Start Training Code at {datetime.now()}')
        sys.stdout.flush()

        # turn on/off log plotting
        self.logplot = kwargs.get('logplot',True)

        # taining details
        self.trainper  = kwargs.get('trainper',0.9)
        self.numepochs = kwargs.get('numepochs',10000)
        self.batchsize = kwargs.get('batchsize',2048)

        # starting learning rate
        self.lr = kwargs.get('lr',1E-3)

        # type of NN
        self.NNtype = kwargs.get('NNtype','MLP_v0')

        # number of nuerons in each layer
        self.H1 = kwargs.get('H1',256)
        self.H2 = kwargs.get('H2',256)
        self.H3 = kwargs.get('H3',256)

        # create list of in labels and out labels
        if 'label_i' in kwargs:
            self.label_i = kwargs['label_i']
        else:
            self.label_i = ['teff','logg','feh','afe','av','rv']
        if 'label_o'in kwargs:
            self.label_o = kwargs['label_o']
        else:
            # assume for roman
            self.label_o = ([
                'roman_wfi_f062','roman_wfi_f087','roman_wfi_f106','roman_wfi_f129','roman_wfi_f146','roman_wfi_f158','roman_wfi_f184','roman_wfi_f213'
                ])

        # number of input and output dimensions
        self.D_in = len(self.label_i)
        self.D_out = len(self.label_o)

        # check for user defined ranges for atm models
        defaultparrange = ({
            'Teff':[0.0,1000000.0],
            'logg':[-2.0,6.0],
            'FeH':[-10.0,10.0],
            'aFe':[-10.0,10.0],
            'Av':[-1.0,100.0],
            'Rv':[2.0,6.0],
            })
        self.parrange = kwargs.get('parrange',defaultparrange)

        self.restartfile = kwargs.get('restartfile',None)
        if self.restartfile is not None:
            print('... Restarting File: {0}'.format(self.restartfile))

        # output hdf5 file name
        self.outfilename = kwargs.get('output','TESTOUT.h5')
        
        # path to CWC phot model grid
        self.modpath = kwargs.get('modpath','./cwc_models.h5')
        
        # the output predictions are normed
        self.norm = kwargs.get('norm',True)
        print(f'... Running with normalized labels: {self.norm}')


        print('... Running Training on Device: {}'.format(device))

        # initialzie class to pull models
        print('... Pulling a first set of models for test set')
        print('... Reading {0:.2f} of grid for test models from {1}'.format(1.0-self.trainper,self.modpath))
        sys.stdout.flush()
        test_mods = readKorg.ReadPhot(
            modpath=self.modpath,
            label_i=self.label_i,
            label_o=self.label_o,
            norm=False,
            returntorch=False,
            type='test',
            trainpercentage=self.trainper,
            parrange=self.parrange,
            verbose=False,
            )
        print(f'... Total number of test models: {len(test_mods)}')        
        test_sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(test_mods),
            batch_size=len(test_mods),
            drop_last=False)

        # test_dataloader = DataLoader(test_mods, batch_size=len(test_mods),shuffle=True)
        test_dataloader = DataLoader(test_mods, sampler=test_sampler,)

        testdata = next(iter(test_dataloader))
        
        # squeeze out the batch dimension since we are only pulling one batch for test
        testdata = np.squeeze(testdata)
                
        self.datacond_in  = np.array([np.where(np.in1d(test_mods.columns,val))[0][0] for val in self.label_i],dtype=int)
        self.datacond_out = np.array([np.where(np.in1d(test_mods.columns,val))[0][0] for val in self.label_o],dtype=int)
                
        self.test_labelsin  = testdata[self.datacond_in,:]
        self.test_labelsout = testdata[self.datacond_out,:]

        print('... Finished reading in test set of models')

        if self.norm:
            self.normfactor = test_mods.normfactor
        else:
            self.normfactor = None
        
        # initialize the output file
        with h5py.File('{0}'.format(self.outfilename),'w') as outfile_i:
            try:
                outfile_i.create_dataset('testlabels_in',
                    data=self.test_labelsin)
                outfile_i.create_dataset('testlabels_out',
                    data=self.test_labelsout)
                outfile_i.create_dataset('label_i',
                    data=np.array([x.encode("ascii", "ignore") for x in self.label_i]))
                outfile_i.create_dataset('label_o',
                    data=np.array([x.encode("ascii", "ignore") for x in self.label_o]))
                if self.norm:
                    for kk in self.label_i:                
                        outfile_i.create_dataset(f'norm_i/{kk}',data=np.array(self.normfactor[kk]))
                    for kk in self.label_o:                
                        outfile_i.create_dataset(f'norm_o/{kk}',data=np.array(self.normfactor[kk]))
            except:
                print('!!! PROBLEM WITH WRITING TO HDF5 !!!')
                raise

        print('... Din: {}, Dout: {}'.format(self.D_in,self.D_out))
        print('... Input Labels: {}'.format(self.label_i))
        print('... Output Labels: {}'.format(self.label_o))
        
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


    def train_mod(self,dryrun=False):
        '''
        function to train the network
        '''
        # start a timer
        starttime = datetime.now()

        if str(device) == 'cuda':
            # determine if this is running within mp
            if len(multiprocessing.current_process()._identity) > 0:
                torch.cuda.set_device(multiprocessing.current_process()._identity[0]-1)

            print('Running on GPU: {0}/{1}'.format(
                torch.cuda.current_device()+1,
                torch.cuda.device_count(),
                ))

        # determine if user wants to start model from old file, or
        # create a new ANN model
        if self.restartfile is not None:
            # create a model
            if os.path.isfile(self.restartfile):
                print('Restarting from File: {0} with NNtype: {1}'.format(self.restartfile,self.NNtype))
                sys.stdout.flush()
                model = photANN.readNN(self.restartfile,nntype=self.NNtype)
            else:
                print('Could Not Find Restart File, Creating a New NN model')
                sys.stdout.flush()
                model = defmod(self.D_in,self.H1,self.H2,self.H3,self.D_out,NNtype=self.NNtype)        
        else:
            # initialize the model
            print('Running New NN with NNtype: {0}'.format(self.NNtype))
            sys.stdout.flush()
            model = defmod(self.D_in,self.H1,self.H2,self.H3,self.D_out,NNtype=self.NNtype)

        print('Model Arch:')
        print(model)

        # set up model to start training
        model.to(device)

        train_mods = readKorg.ReadPhot(
            modpath=self.modpath,
            label_i=self.label_i,
            label_o=self.label_o,
            norm=self.norm,
            returntorch=True,
            type='train',
            trainpercentage=self.trainper,
            parrange=self.parrange,
            )
        
        valid_mods = readKorg.ReadPhot(
            modpath=self.modpath,
            label_i=self.label_i,
            label_o=self.label_o,
            norm=self.norm,
            returntorch=True,
            type='valid',
            trainpercentage=self.trainper,
            parrange=self.parrange,
            )

        train_sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(train_mods),
            batch_size=self.batchsize,
            drop_last=True)
        valid_sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(valid_mods),
            batch_size=self.batchsize,
            drop_last=True)

        train_dataloader = DataLoader(train_mods, sampler=train_sampler,pin_memory=(device.type == "cuda"))
        valid_dataloader = DataLoader(valid_mods, sampler=valid_sampler,pin_memory=(device.type == "cuda"))
        
        nbatches = len(train_dataloader)
        numtrain = nbatches * self.batchsize
        
        print(f'... Number of epochs: {self.numepochs}')
        print(f'... Number of models in each batch: {self.batchsize}')
        print(f'... Number of batches: {nbatches}')
        print(f'... Total Number of training/validation data: {numtrain}')

        # initialize the loss function
        loss_fn = torch.nn.MSELoss(reduction='mean')
        # loss_fn = torch.nn.SmoothL1Loss(beta=0.01)

        # initialize the optimizer
        learning_rate = self.lr
        
        # scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        # optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5, patience=20,
        # )

        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'ln', 'norm']):
                    no_decay.append(param)
                else:
                    decay.append(param)

        optimizer = torch.optim.AdamW(
            [{'params': decay, 'weight_decay': 5e-4},
            {'params': no_decay, 'weight_decay': 0.0}],
            lr=learning_rate
        )

        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, max_lr=5e-3, steps_per_epoch=len(train_dataloader), epochs=self.numepochs
        # )

        fig_loss,ax_loss = plt.subplots(nrows=3,ncols=1,figsize=(7,10),layout='constrained')

        ax_loss[0].set_ylabel('log(Loss per model)')
        ax_loss[0].set_xlim(0,self.numepochs)

        ax_loss[1].set_ylabel('log(Std Residual)')
        ax_loss[1].set_xlim(0,self.numepochs) 

        ax_loss[2].set_ylabel('log(Med Residual)')
        ax_loss[2].set_xlim(0,self.numepochs)

        ax_loss[2].set_xlabel('Epoch')
        
        if dryrun:
            return [model, optimizer, datetime.now()-starttime]
        
        # cycle through epochs
        batchloss_arr = []
        batchloss_std = []
        batchloss_med = []
        validloss_arr = []
        validloss_std = []
        validloss_med = []
        
        for epoch in range(self.numepochs):
            epochtime = datetime.now()

            early_stopper = EarlyStopping(patience=50, min_delta=1e-4, verbose=True)
                
            # train the model
            model.train()            
            batch_loss = 0.0
            running_loss = []
            
            for ii,traindata in enumerate(train_dataloader):

                # read in training and validation data
                # traindata = next(iter(train_dataloader))
                traindata = traindata.squeeze()

                train_labelsin  = traindata[self.datacond_in,:].T
                train_labelsout = traindata[self.datacond_out,:].T

                # create tensor for input training labels
                X_train_Tensor = Variable(train_labelsin.to(device=device, dtype=torch.float32, non_blocking=True))
                # X_train_Tensor = X_train_Tensor.to(device)

                Y_train_Tensor = Variable(train_labelsout.to(device=device, dtype=torch.float32, non_blocking=True), requires_grad=False)
                # Y_train_Tensor = Y_train_Tensor.to(device)

                # compute the model for input training labels
                Y_pred_train_Tensor = model(X_train_Tensor)
                
                # Compute loss and its gradients
                loss = loss_fn(Y_pred_train_Tensor, Y_train_Tensor)

                # print(f"Loss before backward: {loss.item()}")

                # Zero the gradients for every batch
                optimizer.zero_grad()

                # Backpropagation    
                loss.backward()
                # print(f"First grad norm: {model.mlp[0].weight.grad.norm()}")

                # step optimizer to update weights
                optimizer.step()
                # scheduler.step()

                # running_loss += [loss.detach().data.item()]
                running_loss += [loss.item()]

            # normalize by number of batches
            batch_loss = np.sum(running_loss) #/ nbatches
            batchloss_arr.append(batch_loss)
            batchloss_std.append(np.std(running_loss))
            batchloss_med.append(np.median(running_loss))
            
            # evaluate the validation set
            model.eval()
            valid_loss = 0.0
            running_valid = [] 
            with torch.no_grad():
                for ii,validdata in enumerate(valid_dataloader):

                    # read in training and validation data
                    # validdata = next(iter(valid_dataloader))
                    validdata = validdata.squeeze()

                    valid_labelsin  = validdata[self.datacond_in,:].T
                    valid_labelsout = validdata[self.datacond_out,:].T
                    
                    Y_valid_Tensor = Variable(valid_labelsout.to(device=device, dtype=torch.float32, non_blocking=True), requires_grad=False)
                    # Y_valid_Tensor = Y_valid_Tensor.to(device)
                    
                    Y_valid_pred_Tensor = model(Variable(valid_labelsin.to(device=device, dtype=torch.float32, non_blocking=True)))

                    vloss = loss_fn(Y_valid_pred_Tensor, Y_valid_Tensor)
                    # running_valid += [vloss.detach().data.item()]
                    running_valid += [vloss.item()]

            valid_loss = np.sum(running_valid) # / nbatches
            validloss_arr.append(valid_loss)
            validloss_std.append(np.std(running_valid))
            validloss_med.append(np.median(running_valid))

            triggerstop = False
            if early_stopper.step(valid_loss):
                triggerstop = True
            
            # plot the loss curve
            for line in ax_loss[0].lines:
                line.remove()
            ax_loss[0].plot(np.arange(epoch+1),np.log10(batchloss_arr),ls='-',lw=0.5,alpha=0.5,c='C0',label='Training')
            ax_loss[0].plot(np.arange(epoch+1),np.log10(validloss_arr),ls='-',lw=0.5,alpha=0.5,c='C3',label='Validation')

            for line in ax_loss[1].lines:
                line.remove()
            ax_loss[1].plot(np.arange(epoch+1),np.log10(batchloss_std),ls='-',lw=0.5,alpha=0.5,c='C0',label='Training')
            ax_loss[1].plot(np.arange(epoch+1),np.log10(validloss_std),ls='-',lw=0.5,alpha=0.5,c='C3',label='Validation')

            for line in ax_loss[2].lines:
                line.remove()
            ax_loss[2].plot(np.arange(epoch+1),np.log10(batchloss_med),ls='-',lw=0.5,alpha=0.5,c='C0',label='Training')
            ax_loss[2].plot(np.arange(epoch+1),np.log10(validloss_med),ls='-',lw=0.5,alpha=0.5,c='C3',label='Validation')

            fig_loss.savefig('./{0}_loss.png'.format(self.outfilename.replace('.h5','')),dpi=150)

            logcond = ((epoch > 0) and (epoch % 10 == 0)) or triggerstop

            if logcond:

                # After Epoch, write network to output HDF5 file to save progress
                with h5py.File(f'{self.outfilename}','r+') as outfile_i:
                    for kk in model.state_dict().keys():
                        try:
                            del outfile_i['model/{0}'.format(kk)]
                        except KeyError:
                            pass
                        outfile_i.create_dataset(
                            'model/{0}'.format(kk),
                            data=model.state_dict()[kk].cpu().numpy(),
                            compression='gzip')

                # scheduler.step(valid_loss)
                for param_group in optimizer.param_groups:
                    lr_i = param_group['lr']
                
                print(f'... Epoch: {epoch+1} / {self.numepochs} - Training log(Loss): {np.log10(batch_loss):.5f} - Validation log(Loss): {np.log10(valid_loss):.5f} - LR: {lr_i:.2e} - Epoch Time: {(datetime.now()-epochtime)}')
                sys.stdout.flush()

            if triggerstop:
                print('... Early Stopping Triggered')
                break

                
        plt.close(fig_loss)
        torch.cuda.empty_cache()

        print('Finished training model, took: {0}'.format(
            datetime.now()-starttime))
        sys.stdout.flush()
