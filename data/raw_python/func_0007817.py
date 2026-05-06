def nextLunarEclipse(date):
    """ Returns the Datetime of the maximum phase of the
    next global lunar eclipse.

    """

    eclipse = swe.lunarEclipseGlobal(date.jd, backward=False)
    return Datetime.fromJD(eclipse['maximum'], date.utcoffset)