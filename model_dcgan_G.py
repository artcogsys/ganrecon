"""
__author__: K. Seeliger
__status__: Development
__date__: 21-06-2018

Code for the generator (G) of the DCGAN model. 

The complete DCGAN code (i.e. the GAN training code) we used can be found here: https://github.com/musyoku/improved-gan

This is a modular component of the method and can be replaced by any deterministic generative (differentiable) model. 

"""

import chainer.functions as F
from chainer import Chain, ChainList, Variable, report, serializers

from args import args


class GANGenerator(Chain):
    
    def __init__(self, ninput, noutput):
        super(GANGenerator, self).__init__(
            l1 = Linear(args.ndim_z, 512 * 3**2), 
            lb1 = BatchNormalization(512 * 3 ** 2), 
            # TODO: explicit reshape?
            l2 = Deconvolution2D(512, 256, ksize=4, stride=2, pad=0), 
            lb2 = BatchNormalization(256), 
            l3 = Deconvolution2D(256, 128, ksize=4, stride=2, pad=1), 
            lb3 = BatchNormalization(128), 
            l5 = Deconvolution2D(128, 64, ksize=4, stride=2, pad=1), 
            lb5 = BatchNormalization(64), 
            l6 = Deconvolution2D(64, 1, ksize=4, stride=2, pad=1), 
        )
       # TOOD: use_weightnorm=config.use_weightnorm
       # TODO: nobias: false?


    def __call__(self, z):
    
        h = F.relu(self.l1(z))
        h = F.lb1(h)
        
        h = F.relu(self.l2(z))
        h = F.lb2(h)
        
        h = F.relu(self.l3(z))
        h = F.lb3(h)
        
        h = F.relu(self.l4(z))
        h = F.lb4(h)

        h = F.relu(self.l5(z))
        h = F.lb5(h)
        
        img = F.tanh(h)
        
        return img


    def generate_x_from_z(self, z_batch, test=False, as_numpy=False):
        x_batch, _ = self.generator(z_batch, test=test, return_activations=True)
        if as_numpy:  # don't do this when training the linear model
            return self.to_numpy(x_batch)
        return x_batch


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
