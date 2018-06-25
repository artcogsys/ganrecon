# Reconstruct handwritten characters from brains using GANs

Example code for "Generative adversarial networks for reconstructing natural images from brain activity". 

Method for reconstructing images from brain activity with GANs. You need a GAN that is trained for reproducing the target distribution (images that look like your stimuli) and a differentiable method for doing perceptual feature matching (here: layer activations of a convolutional neural network). 

(more notes follow)

Currently a bug in image write or output of generator model (overflowing). 


# Requirements
* Anaconda Python 2.7 version
* `chainer` version 1.24 (install via: `pip install chainer==1.24  --no-cache-dir -vvvv`)
* A GPU for training the feature matching network


# Usage conditions

If you publish using this code or use it in any other way, please cite: 

(preprint) Seeliger, K., Güçlü, U., Ambrogioni, L., Güçlütürk, Y., & van Gerven, M. A. J. (2017). Generative adversarial networks for reconstructing natural images from brain activity. bioRxiv, 226688. https://www.biorxiv.org/content/early/2017/12/08/226688

Please notify the corresponding author in addition. 
