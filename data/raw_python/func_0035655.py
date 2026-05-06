def transitDurationCircular(P, R_s, R_p, a, i):
    r"""Estimation of the primary transit time. Assumes a circular orbit.

    .. math::
        T_\text{dur} = \frac{P}{\pi}\sin^{-1}
        \left[\frac{R_\star}{a}\frac{\sqrt{(1+k)^2 + b^2}}{\sin{a}} \right]

    Where :math:`T_\text{dur}` transit duration, P orbital period,
    :math:`R_\star` radius of the star, a is the semi-major axis,
    k is :math:`\frac{R_p}{R_s}`, b is :math:`\frac{a}{R_*} \cos{i}`

    (Seager & Mallen-Ornelas 2003)
    """

    if i is nan:
        i = 90 * aq.deg

    i = i.rescale(aq.rad)
    k = R_p / R_s  # lit reference for eclipsing binaries
    b = (a * cos(i)) / R_s

    duration = (P / pi) * arcsin(((R_s * sqrt((1 + k) **
                                              2 - b ** 2)) / (a * sin(i))).simplified)

    return duration.rescale(aq.min)