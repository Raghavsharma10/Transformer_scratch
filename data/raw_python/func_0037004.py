def sph_harmonic_tp(nrows, ncols, n, m):
    """Produces Ynm(theta,phi)

        theta runs from 0 to pi and has 'nrows' points.

        phi runs from 0 to 2*pi - 2*pi/ncols and has 'ncols' points.
    
    """
    nuvec = pysphi.ynunm(n, m, n + 1)
    num = nuvec[1:] * (-1) ** m
    num = num[::-1]
    yvec = np.concatenate([num, nuvec])
    
    theta = np.linspace(0.0, np.pi, nrows)
    phi = np.linspace(0.0, 2.0 * np.pi - 2.0 * np.pi / ncols)
    nu = np.array(range(-n, n + 1), dtype=np.complex128)
    
    out = np.zeros([nrows, ncols], dtype=np.complex128)

    for nn in xrange(0, nrows):
        exp_vec = np.exp(1j * nu * theta[nn])
        for mm in xrange(0, ncols):          
            vl = np.sum(yvec * exp_vec)
            out[nn, mm] = 1j ** m * np.exp(1j * m * phi[mm]) * vl

    return out