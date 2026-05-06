def sin_fc(fdata):
    """Apply sine in the Fourier domain."""

    nrows = fdata.shape[0]
    ncols = fdata.shape[1]

    M = nrows / 2
    fdata[int(M - 1), :] = 0
    fdata[int(M + 1), :] = 0
    
    work1 = np.zeros([nrows, ncols], dtype=np.complex128)
    work2 = np.zeros([nrows, ncols], dtype=np.complex128)

    work1[0, :] = fdata[-1, :]
    work1[1:, :] = fdata[0:-1, :]

    work2[0:-1] = fdata[1:, :]
    work2[-1, :] = fdata[0, :]

    fdata[:, :] = 1.0 / (2 * 1j) * (work1 - work2)