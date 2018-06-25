"""
__author__: K. Seeliger
__status__: Development
__date__: 25-06-2018

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

# *.mat format for saving / loading z after training the linear model
from scipy.io import loadmat, savemat

from args import args


class ZWriter(Evaluator):

    """
    Trainer extension for dumping z after each epoch. 
    
    """

    def __init__(self, iterator, target, filename='finalZ.mat'):
        super(ZWriter, self).__init__(iterator, target, device=args.gpu_device)

        self.filen = filename

    def __call__(self, trainer=None):
        
        # some boilerplate code
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
                    
                savemat(self.filen, {'z':pred_z.data})


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

