def prevLunarEclipse(date):
    """ Returns the Datetime of the maximum phase of the
    previous global lunar eclipse.

    """

    eclipse = swe.lunarEclipseGlobal(date.jd, backward=True)
    return Datetime.fromJD(eclipse['maximum'], date.utcoffset)