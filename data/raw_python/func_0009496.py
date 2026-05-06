def maskedConvolve(arr, kernel, mask, mode='reflect'):
    '''
    same as scipy.ndimage.convolve but is only executed on mask==True
    ... which should speed up everything
    '''
    arr2 = extendArrayForConvolution(arr, kernel.shape, modex=mode, modey=mode)
    print(arr2.shape)
    out = np.zeros_like(arr)
    return _calc(arr2, kernel, mask, out)