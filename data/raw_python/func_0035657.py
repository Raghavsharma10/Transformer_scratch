def estimateDistance(m, M, Av=0.0):
    """ estimate the distance to star based on the absolute magnitude, apparent
    magnitude and the absorption / extinction.

    :param m: apparent magnitude
    :param M: absolute magnitude
    :param Av: absorbtion / extinction

    :return: d (distance to object) in parsecs
    """
    try:
        m = float(m)  # basic value checking as there is no units
        M = float(M)
        Av = float(Av)
    except TypeError:
        return np.nan

    d = 10 ** ((m - M + 5 - Av) / 5)

    if math.isnan(d):
        return np.nan
    else:
        return d * aq.pc