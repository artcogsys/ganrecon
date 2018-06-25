"""
__author__: K. Seeliger
__status__: Development
__date__: 21-06-2018


(usage instructions)

"""

from scipy.io import loadmat, savemat


from args import args


# do the reconstruction only here, predicted 


if __name__=="__main__":
    
    loadmat(args.zoutfilen)
    

    imgs = self.trained_gan.generate_x_from_z(z, as_numpy=True)