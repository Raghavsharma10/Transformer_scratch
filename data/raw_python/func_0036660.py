def divsin_fc(fdata):
    """Apply divide by sine in the Fourier domain."""
    
    nrows = fdata.shape[0]
    ncols = fdata.shape[1]

    L = int(nrows / 2)  # Assuming nrows is even, which it should be.
    L2 = L - 2  # This is the last index in the recursion for division by sine.
    
    g = np.zeros([nrows, ncols], dtype=np.complex128)
    g[L2, :] = 2 * 1j * fdata[L - 1, :]

    for k in xrange(L2, -L2, -1):
        g[k - 1, :] = 2 * 1j * fdata[k, :] + g[k + 1, :]

    fdata[:, :] = g