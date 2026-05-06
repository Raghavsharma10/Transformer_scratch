def funcGauss1D(x, mu, sig):
    """ Create 1D Gaussian. Source:
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    arrOut = np.exp(-np.power((x - mu)/sig, 2.)/2)
    # normalize
#    arrOut = arrOut/(np.sqrt(2.*np.pi)*sig)
    # normalize (laternative)
    arrOut = arrOut/np.sum(arrOut)
    return arrOut