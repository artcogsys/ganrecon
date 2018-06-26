"""
__author__: "K. Seeliger"
__status__: "Final"
__date__: "27-06-2018"

Code for the basic linear model that learns to predict the latent space from BOLD data. 

Code for perceptual feature matching. 

It should be possible to improve by using non-linear alternative models. (But simple MLPs like to overfit here.)

"""


import numpy as np

import chainer
from chainer import Chain, ChainList, Variable, report
import chainer.cuda as C
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L

from args import args

import pdb


## Linear regression network model

class LinearRegression(Chain):

    """
    Implements the network for gradient descent-based linear regression. 
    
    """


    def __init__(self, ninput, noutput):
        super(LinearRegression, self).__init__(
            l1 = L.Linear(ninput, noutput, initialW = I.HeNormal()),
        )

    def __call__(self, x):
        # During GAN training, z was randomly drawn as follows: 
        # z_batch = np.random.uniform(-1,1, (batchsize, ndim_z)).astype(np.float32)
        # z_batch = F.normalize(z_batch)

        # We have to normalize in the same way here: 
        y = F.normalize( self.l1(x) )  

        return y



## Regressor model (returning perceptual losses)

class RegressorZ(Chain):

    def __init__(self, predictor, pretrained_gan, featnet = None):
        super(RegressorZ, self).__init__(predictor=predictor)

        self.pretrained_gan = pretrained_gan
        self.featnet = featnet


    def __call__(self, x, img_real):
    
        if type(img_real) != chainer.variable.Variable:  # if validation set
            img_real = Variable(img_real)

        ## Compute latent space from BOLD
        z = self.predictor(x)
        
        ## Generate images from latent space
        img_fake = self.pretrained_gan.generate_img_from_z(z)
        img_fake = F.clip(img_fake, -1.0, 1.0)  # avoid slight overflow of values (after tanh, up to 1.07)

        img_fake.volatile = 'OFF' ; img_real.volatile = 'OFF'  # workaround an issue during validation

        ## Get activations of perceptual features
        _, layer_activations_fake = self.featnet( img_fake, train=False, return_activations=True )
        _, layer_activations_real = self.featnet( img_real, train=False, return_activations=True )
        
        # Note that featnet can also return the non-softmaxed final layer activations (=the classes, here in _ ).
        # Got some bizarre (and no better) results for natural images when also including a class-matching loss. 
        # But (as mentioned in the paper): A loss on higher layers of a convnet trained with a discrete set of 
        # classes (such as ImageNet classes) may *restrict* your reconstructions to these classes, which is 
        # not desired. Computing a loss within a continuous semantic space may be a solution here. 


        ## Compute perceptual losses
        loss = 0.0

        if self.featnet != None:
            for layer_idx in ['pixel'] + args.featn_layers: 

                if layer_idx == 'pixel':   

                    # compute pixel loss l_px
                    loss_px =  args.lambda_pixel * (
                               F.mean_absolute_error( F.resize_images(img_fake, 
                                                           (args.small_img_dims,args.small_img_dims)),
                                                       F.resize_images(img_real, 
                                                           (args.small_img_dims,args.small_img_dims)) )  )
                    loss += loss_px

                else: 

                    layer_idx = int(layer_idx)

                    activ_fake_pos = F.hard_sigmoid( layer_activations_fake[layer_idx]*3.0 - 3.*args.featthre )
                    activ_real_pos = F.hard_sigmoid( layer_activations_real[layer_idx]*3.0 - 3.*args.featthre )
                    # using hard_sigmoid for a differentiable binarization at threshold 1.0

                    if int(layer_idx) == 0:  # negative feature activations only make sense for conv1
                        activ_fake_neg = F.hard_sigmoid( -1.0*layer_activations_fake[layer_idx]*3.0 - 3.*args.featthre )
                        activ_real_neg = F.hard_sigmoid( -1.0*layer_activations_real[layer_idx]*3.0 - 3.*args.featthre )

                        mask_real = (activ_real_pos.data + activ_real_neg.data) > 0

                    else:  # only use positive activations
                        mask_real = activ_real_pos.data > 0
                        loss_pr_neg = 0

                    if np.sum(mask_real[:]) > 0.0:  # if there are any activations above 1.0
                        # compute l_l,m
                        loss_mag =   args.lambda_magnitude * (
                                     F.mean_squared_error(layer_activations_fake[layer_idx][mask_real],   
                                                          layer_activations_real[layer_idx][mask_real])  )

                    else:  # warn and set magnitude loss to 0.0 (does not happen)
                        loss_mag = 0.0
                        print "Warning: No magnitude loss"

                    loss += loss_mag
                 
        report({'loss': loss}, self)

        # Use this code to check whether gradients were computed: 
        #self.predictor.l1.cleargrads()
        #loss.backward()
        #print "Gradients: ", self.predictor.l1.W.grad
        # Do this for all new loss terms. 

        return loss
