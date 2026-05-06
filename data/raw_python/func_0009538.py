def varYSizeGaussianFilter(arr, stdyrange, stdx=0,
                           modex='wrap', modey='reflect'):
    '''
    applies gaussian_filter on input array
    but allowing variable ksize in y
    
    stdyrange(int) -> maximum ksize - ksizes will increase from 0 to given value
    stdyrange(tuple,list) -> minimum and maximum size as (mn,mx)
    stdyrange(np.array) -> all different ksizes in y
    '''
    assert arr.ndim == 2, 'only works on 2d arrays at the moment'
    
    s0 = arr.shape[0]
    
    #create stdys:
    if isinstance(stdyrange, np.ndarray):
        assert len(stdyrange)==s0, '[stdyrange] needs to have same length as [arr]'
        stdys = stdyrange
    else:
        if type(stdyrange) not in (list, tuple):
            stdyrange = (0,stdyrange)
        mn,mx = stdyrange
        stdys  = np.linspace(mn,mx,s0)
    
    #prepare array for convolution:
    kx = int(stdx*2.5)
    kx += 1-kx%2
    ky = int(mx*2.5)
    ky += 1-ky%2
    arr2 = extendArrayForConvolution(arr, (kx, ky), modex, modey)
    
    #create convolution kernels:
    inp = np.zeros((ky,kx))
    inp[ky//2, kx//2] = 1
    kernels = np.empty((s0,ky,kx))
    for i in range(s0):
        stdy = stdys[i]
        kernels[i] = gaussian_filter(inp, (stdy,stdx))

    out = np.empty_like(arr)
    _2dConvolutionYdependentKernel(arr2, out, kernels)
    return out