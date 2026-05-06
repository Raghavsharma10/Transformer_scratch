def troll(roll, dec, v2, v3):
    """ Computes the roll angle at the target position based on::

            the roll angle at the V1 axis(roll),
            the dec of the target(dec), and
            the V2/V3 position of the aperture (v2,v3) in arcseconds.

        Based on the algorithm provided by Colin Cox that is used in
        Generic Conversion at STScI.
    """
    # Convert all angles to radians
    _roll = DEGTORAD(roll)
    _dec = DEGTORAD(dec)
    _v2 = DEGTORAD(v2 / 3600.)
    _v3 = DEGTORAD(v3 / 3600.)

    # compute components
    sin_rho = sqrt((pow(sin(_v2),2)+pow(sin(_v3),2)) - (pow(sin(_v2),2)*pow(sin(_v3),2)))
    rho = asin(sin_rho)
    beta = asin(sin(_v3)/sin_rho)
    if _v2 < 0: beta = pi - beta
    gamma = asin(sin(_v2)/sin_rho)
    if _v3 < 0: gamma = pi - gamma
    A = pi/2. + _roll - beta
    B = atan2( sin(A)*cos(_dec), (sin(_dec)*sin_rho - cos(_dec)*cos(rho)*cos(A)))

    # compute final value
    troll = RADTODEG(pi - (gamma+B))

    return troll