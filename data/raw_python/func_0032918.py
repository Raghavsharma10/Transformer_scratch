def sphericalAngSep(ra0, dec0, ra1, dec1, radians=False):
    """
        Compute the spherical angular separation between two
        points on the sky.

        //Taken from http://www.movable-type.co.uk/scripts/gis-faq-5.1.html

        NB: For small distances you can probably use
        sqrt( dDec**2 + cos^2(dec)*dRa)
        where dDec = dec1 - dec0 and
               dRa = ra1 - ra0
               and dec1 \approx dec \approx dec0
    """

    if radians==False:
        ra0  = np.radians(ra0)
        dec0 = np.radians(dec0)
        ra1  = np.radians(ra1)
        dec1 = np.radians(dec1)

    deltaRa= ra1-ra0
    deltaDec= dec1-dec0

    val = haversine(deltaDec)
    val += np.cos(dec0) * np.cos(dec1) * haversine(deltaRa)
    val = min(1, np.sqrt(val)) ; #Guard against round off error?
    val = 2*np.arcsin(val)

    #Convert back to degrees if necessary
    if radians==False:
        val = np.degrees(val)

    return val