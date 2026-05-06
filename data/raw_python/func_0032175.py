def raDecFromVec(v):
    """
    Taken from
    http://www.math.montana.edu/frankw/ccp/multiworld/multipleIVP/spherical/learn.htm
    Search for "convert from Cartestion to spherical coordinates"

    Adapted because I'm dealing with declination which is defined
    with 90degrees at zenith
    """

    #Ensure v is a normal vector
    v /= np.linalg.norm(v)

    ra_deg=0    #otherwise not in namespace0
    dec_rad = np.arcsin(v[2])
    s = np.hypot(v[0], v[1])
    if s ==0:
        ra_rad = 0
    else:
        ra_rad = np.arcsin(v[1]/s)
        ra_deg = np.degrees(ra_rad)
        if v[0] >= 0:
            if v[1] >= 0:
                pass
            else:
                ra_deg = 360 + ra_deg
        else:
            if v[1] > 0:
                ra_deg = 180 - ra_deg
            else:
                ra_deg = 180 - ra_deg

    raDec = ra_deg, np.degrees(dec_rad)
    return np.array(raDec)