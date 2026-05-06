def ascdiff(decl, lat):
    """ Returns the Ascensional Difference of a point. """
    delta = math.radians(decl)
    phi = math.radians(lat)
    ad = math.asin(math.tan(delta) * math.tan(phi))
    return math.degrees(ad)