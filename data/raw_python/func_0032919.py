def sphericalAngSepFast(ra0, dec0, ra1, dec1, radians=False):
    """A faster (but less accurate) implementation of sphericalAngleSep

    Taken from http://www.movable-type.co.uk/scripts/latlong.html

    For additional speed, set wantSquare=True, and the return value
    is the square of the separation
    """

    if radians==False:
        ra0  = np.radians(ra0)
        dec0 = np.radians(dec0)
        ra1  = np.radians(ra1)
        dec1 = np.radians(dec1)

    deltaRa= ra1-ra0
    deltaDec= dec1-dec0
    avgDec = .5*(dec0+dec1)

    x = deltaRa*np.cos(avgDec)
    val = np.hypot(x, deltaDec)

    if radians == False:
        val = np.degrees(val)

    return val