def dnarcs(decl, lat):
    """ Returns the diurnal and nocturnal arcs of a point. """
    dArc = 180 + 2 * ascdiff(decl, lat)
    nArc = 360 - dArc
    return (dArc, nArc)