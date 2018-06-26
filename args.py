"""
__author__: "K. Seeliger"
__status__: "Final"
__date__: "27-06-2018"

Data set locations. Weight locations. Training parameters. 

Train on CPU: args.gpu_device = -1
Train on GPU #0: args.gpu_device = 0

"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
args = parser.parse_args()


args.debug = False
args.nepochs = 300 if not args.debug else 2 

args.datadir = './data/'
args.featnetdir = './featurematching/'
args.weightsdir = './weights/'
args.outdir = './recon/'

## data and model weights
args.generator_fname = 'weights_DCGAN_G_BRAINS.hdf5'
args.featnet_fname = 'alexgrayBRAINS_model_iter_final'

# stimulus data: trials x pixels  (separate train 'stimTrn' and val 'stimVal')
args.stimulus_fname = args.datadir + 'stim_brains.mat'

# BOLD data: trials x voxel responses  (separate train 'dataTrn' and val 'dataVal')
args.bold_fname = args.datadir + 'bold_brains_SH.mat'  

args.gan_fname = args.weightsdir + 'weights_DCGAN_G_BRAINS.hdf5'


## Training parameters
args.image_dims = 56
args.small_img_dims = 50   # resize dimension (only pixel-wise MSE)

args.calc_pca = True  # whether to do PCA on the data
args.pcavar = 0.99
args.normalize = True   # z-standardize
args.do_weightdecay = True   # leads to 0-image in combination with MLP?
args.l2_lambda = 0.001
args.nbatch = 2

args.gpu_device = -1  # -1: CPU | 0,1: GPU 1,2


## Feature matching hyperparameters 
#  (need to be finetuned, see paper)
args.featthre = 1.0           # feature activation threshold
args.lambda_pixel = 100.0
args.lambda_magnitude = 1.0

args.featn_layers = [0,2,3]  # 0-indexed convnet layers

args.ndim_z = 50  # dimension of random z vector
args.zoutfilen = "finalZ.mat" # name of the z output file


