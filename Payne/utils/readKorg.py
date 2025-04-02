import h5py
import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from torch.utils.data import Dataset, Sampler

class ReadPhot(Dataset):
    """
    Read in Korg BC tables into a torch Dataset object.

    Args:
        Dataset (_type_): _description_

    Kwargs:
        h5path (str): path to HDF5 file that contains the Korg BC tables
        filters (str): which filters to use
        verbose (bool): print out information
    

    """

    def __init__(self, *args, **kwargs):
        super(ReadPhot, self).__init__()

        self.args = args
        self.kwargs = kwargs
        
        self.verbose = self.kwargs.get('verbose',False)

        # path to HDF5 file that contains the Korg BC tables
        self.modpath = self.kwargs.get('modpath',None)
        
        if self.verbose:
            print('... Reading in {0}'.format(self.modpath))
        
        # read in BC tables
        self.h5 = h5py.File(self.modpath,'r')

       # set training boolean, if false then assume test set
        self.datatype = kwargs.get('type','train')
        if self.verbose:
            print(f'... Data Set Type: {self.datatype}')

        # define a RNG with a set seed
        self.rng = np.random.default_rng(42)

        # set if user wants to return a pytorch tensor or a numpy array
        self.returntorch = kwargs.get('returntorch',True)
        if self.verbose:
            print(f'... Return Torch Tensor: {self.returntorch}')

        # set train/test percentage (percentage of MIST that is used for training)
        self.trainper = kwargs.get('trainpercentage',0.9)
        if self.verbose:
            if self.datatype == 'train':
                print(f'... Training Percentage: {self.trainper}')
            if self.datatype == 'test':
                print(f'... Testing Percentage: {1.0-self.trainper}')

        # set if user wants to normalize the input parameters
        self.norm = kwargs.get('norm',True)
        if self.verbose:
            print(f'... Running with Norm: {self.norm}')

        # check for user defined ranges for atm models
        defaultparrange = None
        self.parrange = kwargs.get('parrange',defaultparrange)

        # define input labels
        default_label_i = ([
            'teff',
            'logg',
            'feh',
            'afe',
            'av',
            'rv'
            ])        
        self.label_i = kwargs.get('label_i',default_label_i)
        if self.verbose:
            print('... Input Labels:')
            print(f'{self.label_i}')

        # define output labels (i.e., photometric bands)
        self.label_o = kwargs.get('label_o',None)
        if self.label_o is None:
            # if None, then use all filters
            self.label_o = []
            for kk in self.h5.keys():
                if kk != 'parameters':
                    self.label_o += [f'{kk}_{x}' for x in list(self.h5[kk][()].dtype.names)]
        if self.verbose:
            print('... Output Labels:')
            print(f'{self.label_o}')
        
        # pull input parameters from table
        self.parameters = self.h5['parameters'][()]

        # add column names to self
        self.columns = list(self.parameters.dtype.names)
        self.columns += list(self.label_o)
        
        # compute normaliztion factors for all labels
        self.normfactor = {}
        for ll in self.label_i:
            mid = np.mean(self.parameters[ll])
            std = np.std(self.parameters[ll])
            # self.normfactor[ll] = ([
            #     np.median(self.parameters[ll]),
            #     np.min(self.parameters[ll]),
            #     np.max(self.parameters[ll]),
            #     ])
            self.normfactor[ll] = ([
                mid,
                std,
                ])

            
        for ll in self.label_o:
            f = ll.split('_')[-1]
            ss = ll.replace('_'+f,'')    
            mid = np.mean(self.h5[ss][f][()])
            std = np.std(self.h5[ss][f][()])
            # self.normfactor[ll] = ([
            #     np.median(self.h5[ss][f][()]),
            #     np.min(self.h5[ss][f][()]),
            #     np.max(self.h5[ss][f][()]),
            #     ])
            self.normfactor[ll] = ([
                mid,
                std,
                ])
            
        # add model index to parameter table
        self.parameters = rfn.append_fields(self.parameters,'model_index',
            np.arange(len(self.parameters)),usemask=False)

        # apply parameter ranges
        if self.parrange is not None:
            for ll in self.label_i:
                if ll in self.parrange.keys():
                    if self.verbose:
                        print(f'... Applying parameter range for {ll}: {self.parrange[ll]}')
                    # apply the parameter range
                    self.parameters = self.parameters[self.parameters[ll] >= self.parrange[ll][0]]
                    self.parameters = self.parameters[self.parameters[ll] <= self.parrange[ll][1]]

        # shuffle parameter array
        self.rng.shuffle(self.parameters)
        
        # define test/train/valid split (split non-test data into 70/30 train/valid)
        if self.datatype == 'test':
            self.testind = self.parameters['model_index'][:int(np.rint((1.0-self.trainper) * self.parameters.shape[0]))]
            self.parameters_test = self.parameters[:int(np.rint((1.0-self.trainper) * self.parameters.shape[0]))]
            self.datalen = len(self.testind)
        else:
            trainvalid = self.parameters['model_index'][int(np.rint((1.0-self.trainper) * self.parameters.shape[0])):]
            parameters_trainvalid = self.parameters[int(np.rint((1.0-self.trainper) * self.parameters.shape[0])):]
            if self.datatype == 'train' or self.datatype == 'valid':
                self.trainind = trainvalid[:int(np.rint(0.7*len(trainvalid)))]
                self.parameters_train = parameters_trainvalid[:int(np.rint(0.7*len(trainvalid)))]
                if self.datatype == 'train':
                    self.datalen = len(self.trainind)
                if self.datatype == 'valid':
                    self.validind = trainvalid[int(np.rint(0.7*len(trainvalid))):]
                    self.parameters_valid = parameters_trainvalid[int(np.rint(0.7*len(trainvalid))):]
                    self.datalen = len(self.validind)
            

    # def normf(self,inarr,label):        
    #     return 1.0 + (inarr-self.normfactor[label][0])/(self.normfactor[label][2]-self.normfactor[label][1]) 

    def normf(self, inarr, label):
        mean, std = self.normfactor[label]
        return (inarr - mean) / std

    # def unnormf(self,inarr,label):
    #     return ((inarr-1.0)*(self.normfactor[label][2]-self.normfactor[label][1])) + self.normfactor[label][0]

    def unnormf(self, inarr, label):
        mean, std = self.normfactor[label]
        return (inarr * std) + mean

    def readmod(self,index,filtername):
        """
        Returns a model for index given the filtername
        
        """
        f = filtername.split('_')[-1]
        ss = filtername.replace('_'+f,'')
                
        bc = self.h5[ss][f][index]
        if self.norm:
            bc = self.normf(bc,filtername)
        return bc        
        
    def __len__(self):
        """
        Return the total number of Korg BC rows        

        Returns:
            float : Total rows in Korg BC table
        """
        return self.datalen

    def __getitem__(self, idx):
        """
        Return a draw from the Korg BC table

        Args:
            idx (integer): index integer for row to draw
        """
        # select which set of parameters
        # inpars = self.parameters[idx]
        if self.datatype == 'test':
            selind = self.testind[idx]
            inpars = self.parameters_test[idx]
        elif self.datatype == 'train':
            selind = self.trainind[idx]
            inpars = self.parameters_train[idx]
        elif self.datatype == 'valid':
            selind = self.validind[idx]
            inpars = self.parameters_valid[idx]
            
        # get BC from HDF5 tables
        bcout = []
        for ff in self.label_o:
            bcout.append(self.readmod(selind,ff))

        # build output array with input parametersf
        outarr = [self.normf(inpars[ll],ll) if self.norm else inpars[ll] for ll in self.label_i] + bcout
        outarr = np.array(outarr,dtype=np.float32)
        
        if self.returntorch:
            return torch.tensor(outarr)
        else:
            return outarr
        