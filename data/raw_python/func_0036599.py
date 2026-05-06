def hkm_fc(fdata, Nmax, m, s):
    """ Assume fdata has even rows"""

    f = fdata[:, m]
    L1 = f.size
    MM = int(L1 / 2)
    Q = s.size

    ff = np.zeros(Q, dtype=np.complex128)
    for n in xrange(MM, L1):
        ff[n] = f[n - MM]

    for n in xrange(0, MM):
        ff[n] = f[n + MM]

    # For larger problems, this speeds things up pretty good.
    F = np.fft.fft(ff)
    S = np.fft.fft(s)
    out = 4 * np.pi * np.fft.ifft(F * S)

    return out[0:Nmax + 1]