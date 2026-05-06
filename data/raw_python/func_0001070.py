def qn(phi, *n):
    """
    Calculate the complex flow vector `Q_n`.

    :param array-like phi: Azimuthal angles.

    :param int n: One or more harmonics to calculate.

    :returns:
        A single complex number if only one ``n`` was given or a complex array
        for multiple ``n``.

    """
    phi = np.ravel(phi)
    n = np.asarray(n)

    i_n_phi = np.zeros((n.size, phi.size), dtype=complex)
    np.outer(n, phi, out=i_n_phi.imag)

    qn = np.exp(i_n_phi, out=i_n_phi).sum(axis=1)
    if qn.size == 1:
        qn = qn[0]

    return qn