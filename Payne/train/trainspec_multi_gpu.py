import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau

import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Pool

# from multiprocessing import Pool

import traceback
import numpy as np
import warnings
with warnings.catch_warnings():
     warnings.simplefilter('ignore')
     import h5py
import time,sys,os,glob
from datetime import datetime
try:
     # Python 2.x
     from itertools import imap
except ImportError:
     # Python 3.x
     imap=map

from ..utils.pullspectra import pullspectra

class Net_GPU(nn.Module):  
     def __init__(self, D_in, H, D_out):
          super(Net_GPU, self).__init__()
          self.lin1 = nn.Linear(D_in, H)
          self.lin2 = nn.Linear(H,H)
          self.lin3 = nn.Linear(H,H)
          self.lin4 = nn.Linear(H, D_out)
          # self.lin3 = nn.Linear(H,D_out)
     """
     def forward(self, x):
          x_i = self.encode(x)
          out1 = F.sigmoid(self.lin1(x_i))
          out2 = F.sigmoid(self.lin2(out1))
          # out3 = F.sigmoid(self.lin3(out2))
          # y_i = self.lin4(out3)
          y_i = self.lin3(out2)
          return y_i     
     """
     def forward(self, x):
          x_i = self.encode(x)
          out1 = torch.sigmoid(self.lin1(x_i))
          out2 = torch.sigmoid(self.lin2(out1))
          out3 = torch.sigmoid(self.lin3(out2))
          y_i = self.lin4(out3)
          # y_i = self.lin3(out2)
          return y_i     

     def encode(self,x):
          # convert x into numpy to do math
          x_np = x.data.cpu().numpy()
          try:
               self.xmin
               self.xmax
          except (NameError,AttributeError):
               self.xmin = np.amin(x_np,axis=0)
               self.xmax = np.amax(x_np,axis=0)

          x = (x_np-self.xmin)/(self.xmax-self.xmin)
          return Variable(torch.from_numpy(x).type(dtype))

class TrainSpec_multi_gpu(object):
     """docstring for TrainSpec"""
     def __init__(self, **kwargs):

          # import matplotlib
          # # matplotlib.use('AGG')
          # import matplotlib.pyplot as plt
          # matplotlib.pyplot.ioff()

          # self.plt = plt

          # number of models to train on
          if 'numtrain' in kwargs:
               self.numtrain = kwargs['numtrain']
          else:
               self.numtrain = 20000

          if 'numtest' in kwargs:
               self.numtest = kwargs['numtest']
          else:
               self.numtest = 1000

          # number of iteration steps in training
          if 'niter' in kwargs:
               self.niter = kwargs['niter']
          else:
               self.niter = 200000

          if 'epochs' in kwargs:
               self.epochs = kwargs['epochs']
          else:
               self.epochs = 5

          # number of nuerons in each layer
          if 'H' in kwargs:
               self.H = kwargs['H']
          else:
               self.H = 256

          # wavelength range to train on in angstroms
          if 'waverange' in kwargs:
               self.waverange = kwargs['waverange']
          else:
               self.waverange = [5150.0,5200.0]

          # check for user defined ranges for C3K spectra
          if 'Teff' in kwargs:
               self.Teffrange = kwargs['Teff']
          else:
               self.Teffrange = None

          if 'logg' in kwargs:
               self.loggrange = kwargs['logg']
          else:
               self.loggrange = None

          if 'FeH' in kwargs:
               self.FeHrange = kwargs['FeH']
          else:
               self.FeHrange = None

          if 'aFe' in kwargs:
               self.aFerange = kwargs['aFe']
          else:
               self.aFerange = None

          if 'resolution' in kwargs:
               self.resolution = kwargs['resolution']
          else:
               self.resolution = None

          self.pixpernet = kwargs.get('pixpernet',1)

          if 'verbose' in kwargs:
               self.verbose = kwargs['verbose']
          else:
               self.verbose = False

          # output hdf5 file name
          if 'output' in kwargs:
               self.outfilename = kwargs['output']
          else:
               self.outfilename = 'TESTOUT.h5'

          self.restartfile = kwargs.get('restartfile',False)
          if self.restartfile != False:
               print('... Restarting File: {0}'.format(self.restartfile))

          if 'saveopt' in kwargs:
               self.saveopt = kwargs['saveopt']
          else:
               self.saveopt = False

          if 'logepoch' in kwargs:
               self.logepoch = kwargs['logepoch']
          else:
               self.logepoch = True

          if 'adaptivetrain' in kwargs:
               self.adaptivetrain = kwargs['adaptivetrain']
          else:
               self.adaptivetrain = True

          if 'logdir' in kwargs:
               self.logdir = kwargs['logdir']
          else:
               self.logdir = '.'

          if 'pdfdir' in kwargs:
               self.pdfdir = kwargs['pdfdir']
          else:
               self.pdfdir = '.'
               
          self.MISTpath = kwargs.get('MISTpath',None)
          self.C3Kpath  = kwargs.get('C3Kpath',None)

          # pull C3K spectra for training
          print('... Pulling {0} Testing Spectra and saving labels'.format(self.numtest))
          sys.stdout.flush()
          pullspectra_o = pullspectra(MISTpath=self.MISTpath,C3Kpath=self.C3Kpath)
          spectra_t,labels_t,wavelength_t = pullspectra_o(
               self.numtest,resolution=self.resolution, waverange=self.waverange,
               MISTweighting=True,
               Teff=self.Teffrange,logg=self.loggrange,FeH=self.FeHrange,aFe=self.aFerange)

          self.testlabels = labels_t.tolist()

          self.spectra_o,self.labels_o,self.wavelength = pullspectra_o(
               self.numtest,resolution=self.resolution, waverange=self.waverange,
               MISTweighting=True,
               Teff=self.Teffrange,logg=self.loggrange,FeH=self.FeHrange,aFe=self.aFerange)
          self.spectra = self.spectra_o

          # N is batch size (number of points in X_train),
          # D_in is input dimension
          # H is hidden dimension
          # D_out is output dimension
          self.N = len(self.labels_o)#self.X_train.shape[0]
          try:
               self.D_in = len(self.labels_o[0])#self.X_train.shape[1]
          except IndexError:
               self.D_in = 1
          # self.D_out = self.pixpernet

          print('... Finished Init')
          sys.stdout.flush()

     def __call__(self,pixelarr):
          '''
          call instance so that train_pixel can be called with multiprocessing
          and still have all of the class instance variables

          :params pixelarr:
               Array of pixel numbers that are going to be trained

          '''
          try:
               return self.train_pixel(pixelarr)
          except Exception as e:
               traceback.print_exc()
               print()
               raise e


     def run(self,mp=False,ncpus=1,inpixelarr=[]):
          '''
          function to actually run the training on pixels

          :param mp (optional):
               boolean that turns on multiprocessing to run 
               training in parallel

          '''

          # number of pixels to train
          numspecpixels = self.spectra.shape[1]
          if inpixelarr == []:
               pixellist = range(numspecpixels)
          else:
               pixellist = inpixelarr

          # divide up the pixellist into batches based on pixpernet parameter
          pixelbatchlist = slicebatch(pixellist,self.pixpernet)

          print('... Number of Pixels in Spectrum: {0}'.format(numspecpixels))
          print('... Number of Pixels to Train: {0}'.format(len(pixellist)))
          print('... Training Max {0} pixels per network'.format(self.pixpernet))
          print('... Number of Networks: {0}'.format(len(pixelbatchlist)))
          sys.stdout.flush()

          # turn on multiprocessing if desired
          if mp:
               ##### multiprocessing stuff #######
               try:
                    # determine the number of cpu's and make sure we have access to them all
                    numcpus = open('/proc/cpuinfo').read().count('processor\t:')
                    os.system("taskset -p -c 0-{NCPUS} {PID}".format(NCPUS=numcpus-1,PID=os.getpid()))
               except IOError:
                    pass

               pool = Pool(processes=ncpus)
               # init the map for the pixel training using the pool imap
               mapfn = pool.map

          else:
               # init the map for the pixel training using the standard serial imap
               mapfn = imap

          # start total timer
          tottimestart = datetime.now()

          print('... Starting Training at {0}'.format(tottimestart))
          sys.stdout.flush()

          for pixelbatchlist_i in slicebatch(pixelbatchlist,ncpus):
               try:
                    startpix = pixelbatchlist_i[0].start
                    stoppix  = pixelbatchlist_i[-1].stop
                    wavestart = self.wavelength[startpix:stoppix][0]
                    waveend   = self.wavelength[startpix:stoppix][-1]
                    if stoppix-startpix == 1:
                         print('... Doing Pixel: {0} (w={1})'.format(
                              startpix,wavestart))
                    else:
                         print('... Doing Pixels: {0}-{1} (w={2}-{3})'.format(
                              startpix,stoppix-1,wavestart,waveend))
                    sys.stdout.flush()
               except ValueError:
                    break
               for ii,net in enumerate(mapfn(self,pixelbatchlist_i)):
                    startpix_i = pixelbatchlist_i[ii].start
                    stoppix_i  = pixelbatchlist_i[ii].stop
                    wavestart_i = self.wavelength[startpix_i:stoppix_i][0]
                    waveend_i   = self.wavelength[startpix_i:stoppix_i][-1]

                    outfile_i = h5py.File(
                         '{0}_w{1}_{2}.h5'.format(
                              self.outfilename,wavestart_i,waveend_i
                              ),'w')
                    outfile_i.create_dataset('wavelength',
                         data=self.wavelength[pixelbatchlist_i[ii]])
                    outfile_i.create_dataset('testing_labels'
                         data=np.array(self.testlist)
                         )
                    outfile_i.create_dataset('training_labels',
                         data=np.array(net[3])
                         )

                    try:
                         for kk in net[1].state_dict().keys():
                              outfile_i.create_dataset(
                                   'model_{0}_{1}/model/{2}'.format(
                                        wavestart_i,waveend_i,kk),
                                   data=net[1].state_dict()[kk].cpu().numpy(),
                                   compression='gzip')
                    except RuntimeError:
                         print('!!! PROBLEM WITH WRITING TO HDF5 FOR WAVELENGTH = {0} !!!'.format(wavestart_i))
                         raise

                    outfile_i.close()                  
                    # wave_h5[ii]  = self.wavelength[ii]
                    # self.h5model_write(net[1],outfile,self.wavelength[ii])
                    # if self.saveopt:
                    #    self.h5opt_write(net[2],outfile,self.wavelength[ii])
               # flush output file to save results
               sys.stdout.flush()
               # outfile.flush()
               if stoppix-startpix == 1:
                    print('... Finished Pixel: {0} (w={1}) @ {2}'.format(
                         startpix,wavestart,datetime.now()))
               else:
                    print('... Finished Pixels: {0}-{1} (w={2}-{3}) @ {4}'.format(
                         startpix,stoppix-1,wavestart,waveend,datetime.now()))

          # print out total time
          print('Total time to train network: {0}'.format(datetime.now()-tottimestart))
          sys.stdout.flush()
          # formally close the output file
          # outfile.close()

     # def initout(self,restartfile=False):
     #    '''
     #    function to save all of the information into 
     #    a single HDF5 file
     #    '''

     #    if restartfile == False:
     #         # create output HDF5 file
     #         outfile = h5py.File(self.outfilename,'w', libver='latest', swmr=True)

     #         # add datesets for values that are already defined
     #         label_h5 = outfile.create_dataset('labels',    data=self.labels_o,  compression='gzip')
     #         resol_h5 = outfile.create_dataset('resolution',data=np.array([self.resolution]),compression='gzip')

     #         # define vectorized wavelength array, model array, and optimizer array
     #         wave_h5  = outfile.create_dataset('wavelength',data=np.zeros(len(self.wavelength)), compression='gzip')
     #         # model_h5 = outfile.create_dataset('model_arr', (len(self.wavelength),), compression='gzip')
     #         # opt_h5   = outfile.create_dataset('opt_arr', (len(self.wavelength),), compression='gzip')

     #         outfile.flush()

     #    else:
     #         # read in training file from restarted run
     #         outfile  = h5py.File(restartfile,'r+', libver='latest', swmr=True)

     #         # add datesets for values that are already defined
     #         label_h5 = outfile['labels']
     #         resol_h5 = outfile['resolution']

     #         # define vectorized arrays
     #         wave_h5  = outfile['wavelength']
     #         # model_h5 = outfile['model_arr']
     #         # opt_h5   = outfile['opt_arr']

     #    return outfile,wave_h5

     def train_pixel(self,pixelarr):
          '''
          define training function for each wavelength pixel to run in parallel
          note we create individual neural network for each pixel
          '''

          # start a timer
          starttime = datetime.now()

          startpix = pixelarr[0]
          stoppix  = pixelarr[-1]
          wavestart = self.wavelength[pixelarr][0]
          wavestop  = self.wavelength[pixelarr][-1]

          # determine if this is running within mp
          if len(multiprocessing.current_process()._identity) > 0:
               torch.cuda.set_device(multiprocessing.current_process()._identity[0]-1)

          print('Pixels: {0}-{1} -- Running on GPU: {2}/{3}'.format(
               startpix,stoppix,
               torch.cuda.current_device()+1,
               torch.cuda.device_count(),
               ))

          print('Pixels: {0}-{1}, Wave: {2}-{3}, pulling first spectra'.format(
               startpix,stoppix,wavestart,wavestop))
          pullspectra_i = pullspectra(MISTpath=self.MISTpath,C3Kpath=self.C3Kpath)
          
          # change labels into old_labels
          old_labels_o = self.labels_o

          # create tensor for labels
          X_train_Tensor = Variable(torch.from_numpy(old_labels_o).type(dtype))
          X_train_Tensor = X_train_Tensor.to(device)

          # pull fluxes at wavelength pixel
          Y_train = np.array(self.spectra[:,pixelarr])
          Y_train_Tensor = Variable(torch.from_numpy(Y_train).type(dtype), requires_grad=False)
          Y_train_Tensor = Y_train_Tensor.to(device)

          # determine if user wants to start from old file, or
          # create a new ANN model
          if self.restartfile != False:
               # create a model
               model = readNN(self.restartfile,wavestart,wavestop)
               model.to(device)
          else:
               # determine the acutal D_out for this batch of pixels
               D_out = len(pixelarr)

               # initialize the model
               model = Net_GPU(self.D_in,self.H,D_out)
               model.to(device)

          # set min and max pars to grid bounds for encoding
          model.xmin = np.array([np.log10(2500.0),-1.0,-4.0,-0.2])
          model.xmax = np.array([np.log10(15000.0),5.5,0.5,0.6])

          # initialize the loss function
          loss_fn = torch.nn.MSELoss(reduction='sum')
          # loss_fn = torch.nn.SmoothL1Loss(size_average=False)
          # loss_fn = torch.nn.KLDivLoss(size_average=False)

          # initialize the optimizer
          learning_rate = 0.01
          # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
          optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

          # initialize the scheduler to adjust the learning rate
          scheduler = StepLR(optimizer,3,gamma=0.75)
          # scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.1)

          # initalize the training list
          trainlabels = []


          print('Pixels: {0}-{1}, Wave: {2}-{3}, Start Training...'.format(
               startpix,stoppix,wavestart,wavestop))

          for epoch_i in range(self.epochs): 
               # adjust the optimizer lr
               scheduler.step()
               lr_i = optimizer.param_groups[0]['lr']

               epochtime = datetime.now()

               bestloss = np.inf
               stopcounter = 0

               for t in range(self.niter):
                    steptime = datetime.now()
                    def closure():
                         # Before the backward pass, use the optimizer object to zero all of the
                         # gradients for the variables it will update (which are the learnable weights
                         # of the model)
                         optimizer.zero_grad()

                         # Forward pass: compute predicted y by passing x to the model.
                         y_pred_train_Tensor = model(X_train_Tensor)

                         # Compute and print loss.
                         loss = loss_fn(y_pred_train_Tensor, Y_train_Tensor)

                         # Backward pass: compute gradient of the loss with respect to model parameters
                         loss.backward()
                         
                         if np.isnan(loss.item()):
                              print(y_pred_train_Tensor)
                              print(Y_train_Tensor)

                         if (t+1) % 5000 == 0:
                              print (
                                   '--> WL: {0:6.2f}-{1:6.2f} -- Pix: {2}-{3} -- Ep: {4} -- St [{5:d}/{6:d}] -- Time/step: {7} -- Train Loss: {8:.7f}'.format(
                                   wavestart,wavestop,startpix,stoppix,epoch_i+1,t+1, self.niter, datetime.now()-steptime, loss.item())
                              )
                              sys.stdout.flush()                           

                         return loss

                    # Calling the step function on an Optimizer makes an update to its parameters
                    loss_i = optimizer.step(closure)

                    if np.isnan(loss_i.item()):
                         print(loss_i)
                         print('LOSS ARE NANS')
                         raise ValueError

                    # first allow it to train 10K steps
                    if t > 20000:
                         # check to see if it hits our fit tolerance limit
                         if np.abs(loss_i.item()-bestloss) < 1e-4:
                              stopcounter += 1
                              if stopcounter >=1000:
                                   break
                         else:
                              stopcounter = 0
                         bestloss = loss_i.item()
                              
               # # re-draw spectra for next epoch
               # spectra_o,labels_o,wavelength = pullspectra_i(
               #    self.numtrain,resolution=self.resolution, waverange=self.waverange,
               #    MISTweighting=True,excludelabels=old_labels_o)
               # spectra = spectra_o.T

               spectra_o, labels_o, wavelength = pullspectra_i.pullpixel(
                    pixelarr,num=self.numtrain,resolution=self.resolution, waverange=self.waverange,
                    MISTweighting=True,excludelabels=old_labels_o.T.tolist()+self.testlist,
                    Teff=self.Teffrange,logg=self.loggrange,FeH=self.FeHrange,aFe=self.aFerange
                    )

               # create X tensor
               X_valid = labels_o
               X_valid_Tensor = Variable(torch.from_numpy(labels_o).type(dtype))

               # pull fluxes at wavelength pixel and create tensor
               # Y_valid = np.array(spectra[pixel_no,:]).T
               Y_valid = spectra_o
               Y_valid_Tensor = Variable(torch.from_numpy(Y_valid).type(dtype), requires_grad=False)
               
               # Validation Forward pass: compute predicted y by passing x to the model.
               Y_pred_valid_Tensor = model(X_valid_Tensor)
               Y_pred_valid = Y_pred_valid_Tensor.data.cpu().numpy()

               # calculate the residual at each validation label
               valid_residual = np.squeeze(Y_valid.T-Y_pred_valid.T)
               if valid_residual.ndim == 1:
                    valid_residual = valid_residual.reshape(1,self.numtrain)

               # check to make sure valid_residual isn't all nan's, if so            
               if np.isnan(valid_residual).all():
                    print('Found an all NaN validation Tensor')
                    print('X_valid: ',np.isnan(X_valid).any())
                    print('Y_valid: ',np.isnan(Y_valid).any())
                    print('Y_pred_valid: ',np.isnan(Y_pred_valid).any())
                    raise ValueError

               # create log of the validation step if user wants
               if self.logepoch:                  
                    with open(
                         self.logdir+'/ValidLog_pix{0}_{1}_wave{2}_{3}_epoch{4}.log'.format(
                              startpix,stoppix,wavestart,wavestop,epoch_i+1),
                         'w') as logfile:
                         logfile.write('modnum Teff log(g) [Fe/H] [a/Fe] ')
                         for ww in self.wavelength[pixelarr]:
                              logfile.write('resid_{} '.format(ww))
                         logfile.write('\n')
                         for ii,res in enumerate(valid_residual[0]):
                              logfile.write('{0} '.format(ii+1))
                              logfile.write(np.array2string(X_valid[ii],separator=' ',max_line_width=np.inf).replace('[','').replace(']',''))
                              logfile.write(' ')
                              logfile.write(np.array2string(valid_residual.T[ii],separator=' ',max_line_width=np.inf).replace('[','').replace(']',''))
                              # logfile.write(' {0}'.format(valid_residual.T[ii]))
                              logfile.write('\n')

                    # fig = self.plt.figure()
                    # ax = fig.add_subplot(111)
                    # # residsize = ((10 * 
                    # #  (max(np.abs(valid_residual))-np.abs(valid_residual))/
                    # #  (max(np.abs(valid_residual))-min(np.abs(valid_residual)))
                    # #  )**2.0) + 2.0
                    # for ii,res in enumerate(valid_residual[0]):
                    #    residsize = ((150 * np.abs(valid_residual.T[ii]))**2.0) + 2.0
                    #    scsym = ax.scatter(10.0**X_valid.T[0],X_valid.T[1],s=residsize,alpha=0.5)
                    # lgnd = ax.legend([scsym,scsym,scsym],
                    #    # ['{0:5.3f}'.format(min(np.abs(valid_residual))),
                    #    #  '{0:5.3f}'.format(np.median(np.abs(valid_residual))),
                    #    #  '{0:5.3f}'.format(max(np.abs(valid_residual)))],
                    #    ['0.0','0.5','1.0'],
                    #     loc='upper left',
                    #    )
                    # lgnd.legendHandles[0]._sizes = [2]
                    # lgnd.legendHandles[1]._sizes = [202]
                    # lgnd.legendHandles[2]._sizes = [402]
                    # # ax.invert_yaxis()
                    # # ax.invert_xaxis()
                    # ax.set_xlim(16000,3000)
                    # ax.set_ylim(6,-1.5)
                    # ax.set_xlabel('Teff')
                    # ax.set_ylabel('log(g)')
                    # fig.savefig(
                    #    self.pdfdir+'/ValidLog_pix{0}_{1}_wave{2}_{3}_epoch{4}.png'.format(
                    #         startpix,stoppix,wavestart,wavestop,epoch_i+1),fmt='PNG',dpi=128)
                    # self.plt.close(fig)

               # check if user wants to do adaptive training
               if self.adaptivetrain:
                    # sort validation labels on abs(resid)
                    ind = np.argsort(np.amax(np.abs(valid_residual),axis=0))

                    # determine worst 1% of validation set
                    numbadmod = int(0.01*self.numtrain)
                    # if number of bad models < 5, then by default set it to 5
                    if numbadmod < 5:
                         numbadmod = 5
                    ind_s = ind[-numbadmod:]
                    labels_a = labels_o[ind_s]

                    # determine the number of models to add per new point
                    numselmod = int(0.1*self.numtrain)
                    # if number of new models < 5, then by default set it to 5
                    if numselmod < 5:
                         numselmod = 5

                    numaddmod = numselmod/numbadmod
                    if numaddmod <= 1:
                         numaddmod = 2

                    # make floor of numaddmod == 1
                    # if numaddmod == 0:
                    #    numaddmod = 1


                    # cycle through worst samples, adding 10% new models to training set
                    for label_i in labels_a:
                         nosel = 1
                         epsilon = 0.1
                         newlabelbool = False
                         while True:
                              newlabels = np.array([x+epsilon*np.random.randn(int(numaddmod)) for x in label_i]).T
                              labels_check = pullspectra_i.checklabels(newlabels)
                              # check to make sure labels_ai are unique
                              if all([x_ai not in labels_o.tolist()+self.testlist for x_ai in labels_check.tolist()]):
                                   # print('Pixel: {0}, nosel = {1}'.format(pixel_no+1,nosel))
                                   newlabelbool = True
                                   break
                              # elif (nosel % 100 == 0):
                              #    print('Pixel: {0}, increasing epsilon to {1} at nosel={2}'.format(pixel_no+1,epsilon*3.0,nosel))
                              #    epsilon = epsilon*3.0
                              #    nosel += 1
                              elif (nosel == 100):
                                   # print('Pixel: {0}, could not find new model at nosel={1}, quitting'.format(pixel_no+1,nosel))
                                   # print(newlabels)
                                   break
                              else:
                                   nosel += 1
                         """
                         if newlabelbool:
                              spectra_ai,labels_ai,wavelength = pullspectra_i.selspectra(
                                   newlabels,
                                   resolution=self.resolution, 
                                   waverange=self.waverange,
                                   )
                              Y_valid_a = np.array(spectra_ai.T[pixel_no,:]).T
                         """
                         if newlabelbool:
                              spectra_ai,labels_ai,wavelength = pullspectra_i.pullpixel(
                                   pixelarr,
                                   inlabels=newlabels,
                                   resolution=self.resolution, 
                                   waverange=self.waverange,
                                   )
                              Y_valid_a = spectra_ai
                              Y_valid = np.vstack([Y_valid,Y_valid_a])
                              labels_o = np.append(labels_o,labels_ai,axis=0)
                    X_valid_Tensor = Variable(torch.from_numpy(labels_o).type(dtype))
                    Y_valid_Tensor = Variable(torch.from_numpy(Y_valid).type(dtype), requires_grad=False)

               # re-use validation set as new training set for the next epoch
               old_labels_o = labels_o
               X_train_Tensor = X_valid_Tensor
               Y_train_Tensor = Y_valid_Tensor
               X_train_Tensor = X_train_Tensor.to(device)
               Y_train_Tensor = Y_train_Tensor.to(device)

               # store training labels
               trainlabels = trainlabels + labels_o.tolist()


               print (
                    'Eph [{4:d}/{5:d}] -- WL: {0:.5f}-{1:.5f} -- Pix: {2}-{3} -- Step Time: {6}, LR: {7:.5f}, Valid max(|Res|): {8:.5f}'.format(
                         wavestart, wavestop, startpix,stoppix, epoch_i+1, self.epochs, datetime.now()-epochtime,
                         lr_i,np.nanmax(np.abs(valid_residual)))
                    )
               sys.stdout.flush()            

          print('Trained pixel: {0}-{1}/{2} (wavelength: {3}-{4}), took: {5}'.format(
               startpix,stoppix,len(self.spectra[0,:]),wavestart,wavestop,
               datetime.now()-starttime))
          sys.stdout.flush()

          return [pixelarr, model, optimizer, trainlabels, datetime.now()-starttime]

     # def h5model_write(self,model,th5,wavelength):
     #    '''
     #    Write trained model to HDF5 file
     #    '''
     #    try:
     #         for kk in model.state_dict().keys():
     #              th5.create_dataset('model_{0}/model/{1}'.format(wavelength,kk),
     #                   data=model.state_dict()[kk].numpy(),
     #                   compression='gzip')
     #    except RuntimeError:
     #         print('!!! PROBLEM WITH WRITING TO HDF5 FOR WAVELENGTH = {0} !!!'.format(wavelength))
     #         raise
     #    # th5.flush()

     # def h5opt_write(self,optimizer,th5,wavelength):
     #    '''
     #    Write current state of the optimizer to HDF5 file
     #    '''
     #    for kk in optimizer.state_dict().keys():
     #         # cycle through the top keys
     #         if kk == 'state':
     #              # cycle through the different states
     #              for jj in optimizer.state_dict()['state'].keys():
     #                   for ll in optimizer.state_dict()['state'][jj].keys():
     #                        try:
     #                             # check to see if it is a Tensor or an Int
     #                             data = optimizer.state_dict()['state'][jj][ll].numpy()
     #                        except AttributeError:
     #                             # create an int array to save in the HDF5 file
     #                             data = np.array([optimizer.state_dict()['state'][jj][ll]])
     #                        th5.create_dataset(
     #                             'opt_{0}/optimizer/state/{1}/{2}'.format(wavelength,jj,ll),
     #                             data=data,compression='gzip')
     #         elif kk == 'param_groups':
     #              pgdict = optimizer.state_dict()['param_groups'][0]
     #              for jj in pgdict.keys():
     #                   try:
     #                        th5.create_dataset(
     #                             'opt_{0}/optimizer/param_groups/{1}'.format(wavelength,jj),
     #                             data=np.array(pgdict[jj]),compression='gzip')
     #                   except TypeError:
     #                        th5.create_dataset(
     #                             'opt_{0}/optimizer/param_groups/{1}'.format(wavelength,jj),
     #                             data=np.array([pgdict[jj]]),compression='gzip')

     #    # th5.flush()

def readNN(nnpath,wavestart,wavestop,xmin=None,xmax=None):
     # read in the file for the previous run 
     nnh5file = h5py.File(nnpath,'r')
     nnh5 = nnh5file['model_{0}_{1}'.format(wavestart,wavestop)]

     D_in = nnh5['model/lin1.weight'].shape[1]
     H = nnh5['model/lin1.weight'].shape[0]
     D_out = nnh5['model/lin4.weight'].shape[0]
     model = Net_GPU(D_in,H,D_out)
     model.xmin = xmin#torch.from_numpy(xmin).type(dtype)
     model.xmax = xmax#torch.from_numpy(xmax).type(dtype)

     newmoddict = {}
     for kk in nnh5['model'].keys():
          nparr = np.array(nnh5['model'][kk])
          torarr = torch.from_numpy(nparr).type(dtype)
          newmoddict[kk] = torarr    
     model.load_state_dict(newmoddict)
     model.eval()
     nnh5file.close()
     return model


def slicebatch(inlist,N):
     '''
     Function to slice a list into batches of N elements. Last sublist might have < N elements.
     '''
     return [inlist[ii:ii+N] for ii in range(0,len(inlist),N)]
