def ecef_to_geodetic(x, y, z, method=None):
    """Convert ECEF into Geodetic WGS84 coordinates
    
    Parameters
    ----------
    x : float or array_like
        ECEF-X in km
    y : float or array_like
        ECEF-Y in km
    z : float or array_like
        ECEF-Z in km
    method : 'iterative' or 'closed' ('closed' is deafult)
        String selects method of conversion. Closed for mathematical
        solution (http://www.epsg.org/Portals/0/373-07-2.pdf , page 96 section 2.2.1)
        or iterative (http://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf).
        
    Returns
    -------
    latitude, longitude, altitude
        numpy arrays of locations in degrees, degrees, and km
        
    """

    # quick notes on ECEF to Geodetic transformations 
    # http://danceswithcode.net/engineeringnotes/geodetic_to_ecef/geodetic_to_ecef.html
    
    method = method or 'closed'

    # ellipticity of Earth    
    ellip = np.sqrt(1. - earth_b ** 2 / earth_a ** 2)
    # first eccentricity squared
    e2 = ellip ** 2  # 6.6943799901377997E-3

    longitude = np.arctan2(y, x)
    # cylindrical radius
    p = np.sqrt(x ** 2 + y ** 2)
    
    # closed form solution
    # a source, http://www.epsg.org/Portals/0/373-07-2.pdf , page 96 section 2.2.1
    if method == 'closed':
        e_prime = np.sqrt((earth_a**2 - earth_b**2) / earth_b**2)
        theta = np.arctan2(z*earth_a, p*earth_b)
        latitude = np.arctan2(z + e_prime**2*earth_b*np.sin(theta)**3, p - e2*earth_a*np.cos(theta)**3)
        r_n = earth_a / np.sqrt(1. - e2 * np.sin(latitude) ** 2)
        h = p / np.cos(latitude) - r_n

    # another possibility
    # http://ir.lib.ncku.edu.tw/bitstream/987654321/39750/1/3011200501001.pdf

    ## iterative method
    # http://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf
    if method == 'iterative':
        latitude = np.arctan2(p, z)
        r_n = earth_a / np.sqrt(1. - e2*np.sin(latitude)**2)
        for i in np.arange(6):
            # print latitude
            r_n = earth_a / np.sqrt(1. - e2*np.sin(latitude)**2)
            h = p / np.cos(latitude) - r_n
            latitude = np.arctan(z / p / (1. - e2 * (r_n / (r_n + h))))
            # print h
        # final ellipsoidal height update
        h = p / np.cos(latitude) - r_n

    return np.rad2deg(latitude), np.rad2deg(longitude), h