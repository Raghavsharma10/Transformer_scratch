def subsample(a): # this is more a generic function then a method ...
    """
    Returns a 2x2-subsampled version of array a (no interpolation, just cutting pixels in 4).
    The version below is directly from the scipy cookbook on rebinning :
    U{http://www.scipy.org/Cookbook/Rebinning}
    There is ndimage.zoom(cutout.array, 2, order=0, prefilter=False), but it makes funny borders.
    
    """
    """
    # Ouuwww this is slow ...
    outarray = np.zeros((a.shape[0]*2, a.shape[1]*2), dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):    
            outarray[2*i,2*j] = a[i,j]
            outarray[2*i+1,2*j] = a[i,j]
            outarray[2*i,2*j+1] = a[i,j]
            outarray[2*i+1,2*j+1] = a[i,j]
    return outarray
    """
    # much better :
    newshape = (2*a.shape[0], 2*a.shape[1])
    slices = [slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]