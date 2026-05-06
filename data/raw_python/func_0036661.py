def dphi_fc(fdata):
    """Apply phi derivative in the Fourier domain."""
    
    nrows = fdata.shape[0]
    ncols = fdata.shape[1]
    
    B = int(ncols / 2)  # As always, we assume nrows and ncols are even
    
    a = list(range(0, int(B)))
    ap = list(range(-int(B), 0))
    a.extend(ap)
    
    dphi = np.zeros([nrows, ncols], np.complex128)
    
    for k in xrange(0, nrows):
        dphi[k, :] = a
        
    fdata[:, :] = 1j * dphi * fdata