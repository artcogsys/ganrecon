"""
__author__: "K. Seeliger"
__status__: "Final"
__date__: "27-06-2018"

Trains an AlexNet-like convolutional neural network model for grayscale handwritten character images (1x56x56). 
Used for feature matching purposes only. Can thus return layer activations. 

Nearly identical to alex.py from chainer examples. Model snapshots are written to the current 
directory. 

Uses the data set of 36 handwritten characters (40k examples) from: 
Schomaker, L. and Vuurpijl, L. (2000). Forensic writer identification: A benchmark data set 
and a comparison of two systems [internal report for the Netherlands Forensic Institute].

Van der Maaten, L. (2009). A new benchmark dataset for handwritten character recognition. 
Tilburg University.

"""

from __future__ import print_function

import chainer as chn
from chainer import Variable
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, iterators, serializers
from chainer import cuda
from chainer import Variable
from chainer.training import extensions
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import reporter

import os
import os.path as osp

from fixed_tuple_dataset import TupleDataset  # fixes len vs. size, fixed in later versions of chainer

import numpy as np
import pickle
from matplotlib.image import imsave
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



class Classifier(chn.Chain):
    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t=None, train=True, return_activations=False):

        if train:
            self.y = None
            self.loss = None
            self.accuracy = None
            self.y = self.predictor(x, return_activations)
            self.loss = self.lossfun(self.y, t)
            reporter.report({'loss': self.loss}, self)
            if self.compute_accuracy:
                self.accuracy = self.accfun(self.y, t)
                reporter.report({'accuracy': self.accuracy}, self)
            return self.loss
        else:
            return self.predictor(x, return_activations)


class AlexNet(chn.Chain):

    def __init__(self):
        super(AlexNet, self).__init__(
            conv1 = L.Convolution2D(1, 96, 5, stride=1),
            conv2 = L.Convolution2D(96, 256,  5, stride=1, pad=0),
            conv3 = L.Convolution2D(256, 384,  3, stride=1, pad=0),
            conv4 = L.Convolution2D(384, 384,  3, stride=1, pad=0),
            conv5 = L.Convolution2D(384, 256,  3, stride=1, pad=0),
            fc6 = L.Linear(None, 4096),
            fc7 = L.Linear(4096, 4096),
            fc8 = L.Linear(4096, 36)  # classify 36 handwritten characters
         )

    def __call__(self, x, return_activations=False):

        # Activations for feature matching are returned before applying 
        # any non-linearities.

        activations = []

        h = self.conv1(x)
        if return_activations:
            activations.append(h)  # [0]
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(h)), 3, stride=2)

        h = self.conv2(h)
        if return_activations:
            activations.append(h)  # [1]
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(h)), 3, stride=2)

        h = self.conv3(h)
        if return_activations:
            activations.append(h)  # [2]
        h = F.relu(h)

        h = self.conv4(h)
        if return_activations:
            activations.append(h)  # [3]
        h = F.relu(h)

        h = self.conv5(h)
        if return_activations:
            activations.append(h)  # [4]
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)

        h = self.fc6(h)
        if return_activations:
            activations.append(h)  # [5]
        h = F.dropout(F.relu(h))

        h = self.fc7(h)
        if return_activations:
            activations.append(h)  # [6]
        h = F.dropout(F.relu(h))

        h = self.fc8(h)
        if return_activations:
            activations.append(h)  # [7]
        if return_activations:
            return h, activations  # return non-softmax model output and activations

        return h  # only return model output (non-softmax)


if __name__ == "__main__":

    ## parameters
    gpu_id = -1   # -1 CPU | 0,1 GPU 1,2
    batchsize = 128
    train_epochs = 1000
    dims = 56
    col_n = 1  # color dimension
    val_interval = ( 5000, 'iteration')  # epochs until extensions are triggered
    log_interval = ( 1000, 'iteration')

    datafilen = 'handwrittencharacters.mat'  # key 'X': 40134x3136  |  key 'labels': letters a-z and digits 1-10

    ## Load handwritten character data
    data_file = loadmat(datafilen) 

    labels = data_file['labels']
    le = preprocessing.LabelEncoder()  # letters a-z and digits 1-10 --> 36 integer indices
    le.fit(labels)
    labels = le.transform(labels)

    ## Reshape to 2D and 1 channel for chainer (transpose due to col vs. row major order)
    img_set = np.reshape(data_file['X'], [data_file['X'].shape[0], 1, 56, 56]).transpose([0, 1, 3, 2]).astype('float32')  

    assert(img_set.shape[0] == labels.shape[0])  # sanity check

    ## Move to range [-1.0,1.0]
    img_set = (img_set - 0.5) * 2.0

    ## Create training data set and iterators
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(img_set, labels, test_size = 0.1)
    
    assert(train_imgs.shape[1:] == val_imgs.shape[1:])   # sanity check
    
    train_data = TupleDataset(train_imgs, np.array(train_labels, np.int32))
    val_data = TupleDataset(val_imgs, np.array(val_labels, np.int32))

    train_iter = iterators.SerialIterator(train_data, batchsize, shuffle=True)
    val_iter = iterators.SerialIterator(val_data, batchsize, repeat=False, shuffle=False)

    ## Create model
    optimizer = chn.optimizers.MomentumSGD(lr=0.001, momentum=0.4)
    model = L.Classifier(AlexNet())
    optimizer.setup(model)

    ## Use GPU if specified
    if gpu_id > -1: 
        chn.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()

    ## Set up trainer and updater
    updater = chn.training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = chn.training.Trainer(updater, (train_epochs, 'epoch'))
   
    ## Set up trainer extensions and reports    
    trainer.extend(extensions.Evaluator(val_iter, model, device=gpu_id), 
                   trigger=val_interval)
    trainer.extend(extensions.snapshot_object(model, 'alexgrayBRAINS_model_iter_{.updater.iteration}'), 
                   trigger=val_interval)
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy']), 
                   trigger=log_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    
    ## Run training
    trainer.run()
