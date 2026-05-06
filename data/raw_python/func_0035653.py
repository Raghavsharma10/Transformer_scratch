def ratioTerminatorToStar(H_p, R_p, R_s):  # TODO add into planet class
    r"""Calculates the ratio of the terminator to the star assuming 5 scale
    heights large. If you dont know all of the input try
    :py:func:`calcRatioTerminatorToStar`

    .. math::
        \Delta F = \frac{10 H R_p + 25 H^2}{R_\star^2}

    Where :math:`\Delta F` is the ration of the terminator to the star,
    H scale height planet atmosphere, :math:`R_p` radius of the planet,
    :math:`R_s` radius of the star

    :param H_p:
    :param R_p:
    :param R_s:
    :return: ratio of the terminator to the star
    """

    deltaF = ((10 * H_p * R_p) + (25 * H_p**2)) / (R_s**2)
    return deltaF.simplified