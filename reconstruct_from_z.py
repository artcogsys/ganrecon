"""
__author__: "K. Seeliger"
__status__: "Final"
__date__: "27-06-2018"

Generates and writes validation set images based on the predicted z saved in args.outdir+args.zoutfilen . 

"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import numpy as np

from model_dcgan_G import GANGenerator
from scipy.io import loadmat, savemat

from utilities import write_images_png
from train_linear_model import load_stim_data

from args import args


if __name__=="__main__":

    print "Loading z and original images..."
    z_val = loadmat(args.outdir+args.zoutfilen)['z'].astype(np.float32)
    _, orig_imgs_val = load_stim_data(args)

    print "Loading Generator..."
    dcgan = GANGenerator()
    dcgan.load_weights_from_hdf5(args.gan_fname)

    print "Generating images..."
    recon_imgs_val = dcgan.generate_img_from_z(z_val, as_numpy=True)

    print "Writing images..."
    write_images_png(recon_imgs_val, orig_imgs_val, args.outdir)