import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import nnx

import warnings
import h5py
import time,sys,os,glob
from datetime import datetime

class Net(object):
    def __init__(self, nnpath=None,nntype='MLP_v1',normed=True):
        self.normed = normed
        self.readNN(nnpath=nnpath,nntype=nntype)

    def readNN(self,nnpath=None,nntype='MLP_v1'):
        # read in normalization info
        nnh5 = h5py.File(nnpath,'r')

        self.label_i = [x.decode('utf-8') for x in nnh5['label_i'][()]]
        self.label_o = [x.decode('utf-8') for x in nnh5['label_o'][()]]

        if self.normed:
            self.norm_i = [nnh5[f'norm_i/{kk}'][()] for kk in self.label_i]
            self.norm_o = [nnh5[f'norm_o/{kk}'][()] for kk in self.label_o]

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

        nnh5.close()

    def evalMLP(self,x):

        x_i = jnp.copy(jnp.asarray(x))        

        if len(x.shape) == 1:
            if self.normed:
                x_ii = jnp.zeros(x.shape,dtype=float)
                for ii,n_i in enumerate(self.norm_i):
                    x_ii = x_ii.at[ii].set((x_i[ii]-n_i[0])/n_i[1])
        else:
            if self.normed:
                x_ii = jnp.zeros(x.shape,dtype=float)                
                for ii,n_i in enumerate(self.norm_i):
                    x_ii = x_ii.at[:,ii].set((x_i[:,ii]-n_i[0])/n_i[1])


        y_i = self.mlp(x_ii)

        if self.normed:
            if len(x.shape) == 1:
                y = jnp.zeros(y_i.shape,dtype=float)
                for ii,n_i in enumerate(self.norm_o):
                    y = y.at[ii].set((y_i[ii]-n_i[0])/n_i[1])
            else:
                y = jnp.zeros(y_i.shape,dtype=float)
                for ii,n_i in enumerate(self.norm_o):
                    y = y.at[:,ii].set((y_i[:,ii]-n_i[0])/n_i[1])
        else:
            y = y_i

        return y        


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
