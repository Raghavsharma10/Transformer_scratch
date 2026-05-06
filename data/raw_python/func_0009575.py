def _coarsenImage(image, f):
    '''
    seems to be a more precise (but slower)
    way to down-scale an image
    '''
    from skimage.morphology import square
    from skimage.filters import rank
    from skimage.transform._warps import rescale
    selem = square(f)
    arri = rank.mean(image, selem=selem)
    return rescale(arri, 1 / f, order=0)