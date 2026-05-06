def mindx(m, nmax, mmax):
    """index to the first n value for a give m within the spherical 
    coefficients vector. Used by sc_to_fc"""

    ind = 0
    NN = nmax + 1

    if np.abs(m) > mmax:
        raise Exception("|m| cannot be larger than mmax")

    if (m != 0):
        ind = NN
        ii = 1
        for i in xrange(1, np.abs(m)):
            ind = ind + 2 * (NN - i)
            ii = i + 1

        if m > 0:
            ind = ind + NN - ii

    return ind