def vspht(vsphere, nmax=None, mmax=None):
    """Returns a VectorCoefs object containt the vector spherical harmonic
    coefficients of the VectorPatternUniform object"""
    
    if nmax == None:
        nmax = vsphere.nrows - 2 
        mmax = int(vsphere.ncols / 2) - 1
    elif mmax == None:
        mmax = nmax

    if mmax > nmax:
        raise ValueError(err_msg['nmax_g_mmax'])

    if nmax >= vsphere.nrows - 1:
        raise ValueError(err_msg['nmax_too_lrg'])

    if mmax >= vsphere.ncols / 2:
        raise ValueError(err_msg['mmax_too_lrg'])

    dnrows = vsphere._tdsphere.shape[0]
    ncols = vsphere._tdsphere.shape[1]

    if np.mod(ncols, 2) == 1:
        raise ValueError(err_msg['ncols_even'])
        
    ft = np.fft.fft2(vsphere._tdsphere) / (dnrows * ncols)
    ops.fix_even_row_data_fc(ft)
    
    ft_extended = np.zeros([dnrows + 2, ncols], dtype=np.complex128)
    ops.pad_rows_fdata(ft, ft_extended)
    
    pt = np.fft.fft2(vsphere._pdsphere) / (dnrows * ncols)
    ops.fix_even_row_data_fc(pt)
    
    pt_extended = np.zeros([dnrows + 2, ncols], dtype=np.complex128)
    ops.pad_rows_fdata(pt, pt_extended)
    
    ftmp = np.copy(ft_extended)
    ptmp = np.copy(pt_extended)
    Lf1 = ops.sinLdot_fc(ft_extended, pt_extended)
    Lf2 = ops.sinLdot_fc(-1j * ptmp, 1j * ftmp)
    
    # check if we are using c extended versions of the code or not
    if use_cext: 
        N = nmax + 1;
        NC = N + mmax * (2 * N - mmax - 1);
        sc1 = np.zeros(NC, dtype=np.complex128)
        sc2 = np.zeros(NC, dtype=np.complex128)
        csphi.fc_to_sc(Lf1, sc1, nmax, mmax)
        csphi.fc_to_sc(Lf2, sc2, nmax, mmax)
    else:   
        sc1 = pysphi.fc_to_sc(Lf1, nmax, mmax)
        sc2 = pysphi.fc_to_sc(Lf2, nmax, mmax)

    vcoefs = VectorCoefs(sc1, sc2, nmax, mmax)

    nvec = np.zeros(nmax + 1, dtype=np.complex128)

    for n in xrange(1, nmax + 1):
        nvec[n] = 1.0 / np.sqrt(n * (n + 1.0))

    vcoefs.scoef1.window(nvec)
    vcoefs.scoef2.window(nvec)
        
    return vcoefs