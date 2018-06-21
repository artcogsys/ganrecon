"""
__author__: K. Seeliger
__status__: Development
__date__: 21-06-2018


(usage instructions)

"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, time

from chainer import training, datasets, iterators, serializers, optimizers, optimizer, cuda
from chainer.training import extensions

from scipy.io import loadmat
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

from model_linear import RegressorZ, LinearRegression
from model_dcgan_G import generator
from recon_utilities import *
from featurematching.train_featurematching_handwritten import AlexNet, Classifier

from args import args



def load_bold_data(args): 

    print "Loading BOLD data..."
    bold_f = loadmat(args.bold_fname)
    
    bold_trn = bold_f['dataTrn']
    bold_val = bold_f['dataVal']
    
    n_voxels = bold_trn.shape[0]
    
    print "Standardizing..."
    if args.normalize:   # do z standardization
        scaler = StandardScaler(with_std=False)
        scaler.fit(bold_trn) 
        bold_trn = scaler.transform(bold_trn)
        bold_val = scaler.transform(bold_val)

    if args.calc_pca: 
        print "Starting PCA..."
        pca = decomposition.PCA(args.pcavar)
        pca.fit(bold_trn)

        print "PCA computed. Applying to data sets..."
        bold_trn = pca.transform( bold_trn ).astype('float32')
        bold_val = pca.transform( bold_val ).astype('float32')

        n_vox_pca = bold_trn.shape[1]

        print "Number principal components used:", n_vox_pca
        
    return bold_trn, bold_val



def load_stim_data(args):

    stim_f = loadmat(args.stimulus_file)

    stim_trn = stim_f['stimTrn']
    stim_val = stim_f['stimVal']
    
    # GAN produces range [-1.0, 1.0], so change range: 
    stim_trn = (stim_trn/255.0 * 2.0 - 1.0).astype('float32')
    stim_val = (stim_val/255.0 * 2.0 - 1.0).astype('float32')

    # add singleton color dimension for chainer
    stim_trn = stim_trn[:,np.newaxis,:,:]
    stim_val = stim_val[:,np.newaxis,:,:]

    return stim_trn, stim_val



if __name__ == "__main__":

    ## Load data
    print "Loading BOLD data..."
    bold_trn, bold_val = load_bold_data(args)

    print "Loading stimulus data..."
    stim_trn, stim_val = load_stim_data(args)
    
    print "Number of training samples:", bold_trn.shape[0]
    print "Number of validation samples:", bold_val.shape[0]
    
    bold_vox_dim = bold_trn.shape[1]
    
    # Sanity checks for NaNs:
    assert(~np.isnan(np.sum(stim_trn[:]))) ; assert(~np.isnan(np.sum(stim_val[:])))
    assert(~np.isnan(np.sum(bold_trn[:]))) ; assert(~np.isnan(np.sum(bold_val[:])))


    ## Load models
    print "Building feature matching model and loading pretrained weights..."
    # This can be replaced with other differentiable perceptual feature extraction methods. 
    featnet_fn = args.featnetdir + args.featnet_fname
    if args.gpu_device != -1:  
        alexnet = Classifier(AlexNet()).to_gpu(device=args.gpu_device)
    else: 
        alexnet = Classifier(AlexNet())
    serializers.load_npz(featnet_fn, alexnet)
    
    print "Building G and loading pretrained weights..."
    # TODO


    ## Prepare training
    print "Building datasets and model trainer..."
    train = datasets.tuple_dataset.TupleDataset(bold_trn, stim_trn)
    validation = datasets.tuple_dataset.TupleDataset(bold_val, stim_val)
    train_iter = iterators.SerialIterator(train, batch_size=args.nbatch, repeat=True, shuffle=True)
    validation_iter = iterators.SerialIterator(validation, batch_size=bold_val.shape[0], repeat=False, shuffle=False)

    linearmodel = RegressorZ( LinearRegression(bold_vox_dim, args.ndim_z), 
                              pretrained_gan=dcgan, featnet=alexnet )

    # Set up optimizer
    optim = optimizers.Adam()
    optim.setup(linearmodel)

    if args.do_weightdecay:
        optim.add_hook(optimizer.WeightDecay(args.l2_lambda))

    updater = training.StandardUpdater(train_iter, optim, device=args.gpu_device)

    # Set up trainer and extensions
    trainer = training.Trainer(updater, (args.nepochs, 'epoch'), out=args.outdir)
    trainer.extend(extensions.Evaluator(validation_iter, model, device=args.gpu_device))
    trainer.extend(extensions.LogReport(log_name='linearmodel_train.log'))
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']) )
    trainer.extend(extensions.ProgressBar())
    
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss_plotreport.png', trigger=(1, 'epoch')))
    # Save model snapshots 
    trainer.extend(extensions.snapshot_object(linearmodel, 'reconstructionmodel_{.updater.iteration}'), 
                   trigger=(20, 'epoch'))


    ## Run training
    trainer.run()
