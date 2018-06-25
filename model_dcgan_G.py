"""
__author__: K. Seeliger
__status__: Development
__date__: 21-06-2018

Code for the generator (G) of the DCGAN model. 

The complete DCGAN code (i.e. the GAN training code) we used can be found here: https://github.com/musyoku/improved-gan

This is a modular component of the method and can be replaced by any deterministic generative (differentiable) model. 

"""

import chainer.functions as F
import chainer.links as L
from chainer import Chain, ChainList, Variable, report, serializers

import os

from args import args

import pdb


class GANGenerator(Chain):
    
    def __init__(self):
        super(GANGenerator, self).__init__(
            link_0 = L.Linear(args.ndim_z, 2048), 
            link_2 = L.BatchNormalization(2048), 
            link_4 = L.Deconvolution2D(512, 256, ksize=4, stride=2, pad=0), 
            link_5 = L.BatchNormalization(256), 
            link_7 = L.Deconvolution2D(256, 128, ksize=4, stride=2, pad=0), 
            link_8 = L.BatchNormalization(128), 
            link_10 = L.Deconvolution2D(128, 64, ksize=4, stride=2, pad=1), 
            link_11 = L.BatchNormalization(64), 
            link_13 = L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1, 
                                        outsize=[args.image_dims, args.image_dims]), 
        )
       # TOOD: use_weightnorm=config.use_weightnorm
       # TODO: nobias: false in all deconv layers
       # TODO: use_gamma, use_beta, use_cudnn und use_beta in batchnorm
       # TODO: decay 0.9 in batchnorm, eps 2e-05 (check again whether default)
       # TODO: use_gamma und u


    def __call__(self, z):
    
        h = F.relu(self.link_0(z))
        h = self.link_2(h)
        
        h = F.reshape(h, (z.shape[0], 512, 2, 2) )
        
        h = F.relu(self.link_4(h))
        h = self.link_5(h)
        
        h = F.relu(self.link_7(h))
        h = self.link_8(h)
        
        h = F.relu(self.link_10(h))
        h = self.link_11(h)

        img = F.tanh(self.link_13(h))
        
        return img


    def generate_img_from_z(self, z_batch, test=False, as_numpy=False):
        img_batch = self.__call__(z_batch)
        if as_numpy:  # don't do this when training the linear model
            return self.to_numpy(img_batch)
        return img_batch


    def load_weights_from_hdf5(self, filename):
		if os.path.isfile(filename):
			print "Loading generator weights from {} ...".format(filename)
			serializers.load_hdf5(filename, self)
		else:
			print "Error: Filename", filename, "not found. "

        
    def sample_z(self, batchsize=1):   # sample z (for GAN training)
        ndim_z = args.ndim_z
        
        z_batch = np.random.uniform(-1,1, (batchsize, ndim_z)).astype(np.float32)
        z_batch = F.normalize(z_batch).data
        
        return z_batch
