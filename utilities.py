"""
__author__: "K. Seeliger"
__status__: "Final"
__date__: "27-06-2018"

Utilities for writing reconstructions from a trained linear model. 

"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import numpy as np
import copy

from chainer.dataset import iterator
from chainer import reporter as reporter_module
from chainer.variable import Variable
from chainer.training.extensions.evaluator import Evaluator

from scipy.io import loadmat, savemat
import pdb

from args import args


class ZWriter(Evaluator):

    """
    Trainer extension for dumping z after each epoch. 
    
    """

    def __init__(self, iterator, target, filename='finalZ.mat'):
        super(ZWriter, self).__init__(iterator, target, device=args.gpu_device)

        self.filen = filename

    def __call__(self, trainer=None):

        iterator = self._iterators['main']
        linearmodel = self._targets['main'].predictor

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                in_vars = tuple( Variable(x) for x in in_arrays )

                bold = in_vars[0]
                pred_z = linearmodel(bold).data

                if args.gpu_device != -1: 
                    pred_z = pred_z.get()
                    
                savemat(self.filen, {'z':pred_z})



def write_images_png(recon_imgs_val, orig_imgs_val, outdir):

    if np.max(recon_imgs_val[:])>1.0:   # sanity check, can happen (infrequently)
        print "Out of bounds values encountered in reconstruction image. Clipping..", np.max(recon_imgs_val[:])
                
    # move back between 0 and 1
    recon_imgs_val = np.clip(  np.squeeze((recon_imgs_val + 1.0) / 2.0), 0.0, 1.0 ) 
    orig_imgs_val = np.squeeze((orig_imgs_val + 1.0) / 2.0)

    n = orig_imgs_val.shape[0]
    f, ax = plt.subplots(int( np.ceil(n/5.0)*2 ), 5, figsize=(40,80))

    draw_i = 0

    # Plot images and their reconstructions in rows of 5
    for row in xrange(0, int( np.ceil(n/5.0)*2 ),2):
        for i in xrange(5): 
            if draw_i>=n:
                ax[row , i].axis('equal')
                ax[row , i].axis('off')
                ax[row+1 , i].axis('equal')
                ax[row+1 , i].axis('off')
                draw_i += 1
                continue

            ax[row , i].imshow(np.squeeze(orig_imgs_val[(row/2)*5 + i]), cmap='gray', vmin=0.0, vmax=1.0)
            ax[row , i].axis('equal')
            ax[row , i].axis('off')

            ax[row+1 , i].imshow(np.squeeze(recon_imgs_val[(row/2)*5 + i]), cmap='gray', vmin=0.0, vmax=1.0)
            ax[row+1 , i].axis('equal')
            ax[row+1 , i].axis('off')

            draw_i += 1

    plt.savefig(outdir + 'recons.png')
    plt.close()



class FiniteIterator(iterator.Iterator,):

    """
    Dataset iterator that reads the examples [0:batch_size] in serial order.
    
    """

    def __init__(self, dataset, batch_size, shuffle = False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self.shuffle = shuffle

    def __next__(self):

        if self.epoch > 0:
            raise StopIteration

        N = len(self.dataset)

        if self.shuffle: 
            rand_selection = (np.random.choice(np.arange(len(self.dataset)), size=self.batch_size)).tolist()
            batch = self.dataset[:]
            batch = [batch[i] for i in rand_selection]
        else: 
            batch = self.dataset[0:self.batch_size]

        self.epoch += 1
        self.is_new_epoch = True

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)

