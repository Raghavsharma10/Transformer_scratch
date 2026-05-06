def s_data(nrows_fdata, Nmax, Q):
    """ I am going to assume we will always have even data. This is pretty 
    safe because it means that we have measured both poles of the sphere and 
    have data that has been continued.

        nrows_fdata:  Number of rows in fdata.
        Nmax:         The largest number of n values desired.
        Q:            A value greater than nrows_fdata + Nmax. This can be
                      selected to be factorable into small primes to 
                      increase the speed of the fft (probably not that big 
                      of a deal today).

    """

    if np.mod(nrows_fdata, 2) == 1:
        raise Exception("nrows_fdata must be even.")
    
    L1 = nrows_fdata

    s = np.zeros(Q, dtype=np.complex128)
    MM = int(L1 / 2)

    for nu in xrange(-MM, MM + Nmax + 1):
        if np.mod(nu, 2) == 1:
            s[nu - MM] = -1j / nu

    return s