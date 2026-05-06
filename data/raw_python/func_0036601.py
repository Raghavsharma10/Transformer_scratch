def sc_to_fc(spvec, nmax, mmax, nrows, ncols):
    """assume Ncols is even"""

    fdata = np.zeros([int(nrows), ncols], dtype=np.complex128)

    for k in xrange(0, int(ncols / 2)):
        if k < mmax:
            kk = k
            ind = mindx(kk, nmax, mmax)
            vec = spvec[ind:ind + nmax - np.abs(kk) + 1]
            fdata[:, kk] = fcvec_m_sc(vec, kk, nmax, nrows)

            kk = -(k + 1)
            ind = mindx(kk, nmax, mmax)
            vec = spvec[ind:ind + nmax - np.abs(kk) + 1]
            fdata[:, kk] = fcvec_m_sc(vec, kk, nmax, nrows)

        if k == mmax:
            kk = k
            ind = mindx(kk, nmax, mmax)
            vec = spvec[ind:ind + nmax - np.abs(kk) + 1]
            fdata[:, kk] = fcvec_m_sc(vec, kk, nmax, nrows)

    return fdata