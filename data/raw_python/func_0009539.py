def numbaGaussian2d(psf, sy, sx):
    '''
    2d Gaussian to be used in numba code
    '''
    ps0, ps1 = psf.shape
    c0,c1 = ps0//2, ps1//2
    ssx = 2*sx**2
    ssy = 2*sy**2
    for i in range(ps0):
        for j in range(ps1):
            psf[i,j]=exp( -( (i-c0)**2/ssy
                            +(j-c1)**2/ssx) )
    psf/=psf.sum()