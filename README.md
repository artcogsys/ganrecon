Reconstruct handwritten characters from brains using GANs
=========================================================

Example code for the paper "Generative adversarial networks for reconstructing
natural images from brain activity".

Method for reconstructing images from brain activity with GANs. You need a GAN
that is trained for reproducing the target distribution (images that look like
your stimuli) and a differentiable method for doing perceptual feature matching
(here: layer activations of a convolutional neural network).

The method uses linear regression implemented as a neural network to predict the
latent space `z`. Losses are calculated in image space and backpropagated
through the loss terms and the GAN over `z` to the weights of the linear
regression layer.


Usage notes
===========

... for the handwritten characters example:

1.  Run `train_linear_model.py`, preferably on a GPU. This will produce
    `./recon/finalZ.mat`  which contains z predictions on your validation set.

2.  Run `reconstruct_from_z.py` to generate a PNG with reconstructions of the
    validation data in `./recon/recons.png`.

... for your own data:

1.  Train a GAN for your stimulus domain (e.g. natural grayscale images of size
    [64 64]). During training z should be drawn from a uniform distribution in
    [-1 1] and normalized (see `sample_z()` in `model_dcgan_G.py`).

2.  Train a differentiable network for feature matching. The training code for
    the AlexNet used for handwritten digits can be found in
    `./featurematching/train_featurematching_handwritten.py`.

3.  Adapt some parameters in `args.py` and  `train_linear_model.py` (and
    hopefully little of the rest). Fine-tune the weights for the loss terms on
    an isolated data set.

4.  You should be able to just run `train_linear_model.py` then.


Requirements
============

-   Anaconda Python 2.7 version

-   `chainer` version 1.24 (install via: `pip install chainer==1.24
    --no-cache-dir -vvvv`)

-   A GPU for training the feature matching network


Usage conditions
================

If you publish using this code or use it in any other way, please cite:

(preprint) Seeliger, K., Güçlü, U., Ambrogioni, L., Güçlütürk, Y., & van Gerven,
M. A. J. (2017). Generative adversarial networks for reconstructing natural
images from brain activity. bioRxiv, 226688.
https://www.biorxiv.org/content/early/2017/12/08/226688

Please notify the corresponding author in addition.
