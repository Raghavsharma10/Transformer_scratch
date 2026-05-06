def from_jd(jd, method=None):
    '''Calculate date in the French Revolutionary
    calendar from Julian day.  The five or six
    "sansculottides" are considered a thirteenth
    month in the results of this function.'''
    method = method or 'equinox'

    if method == 'equinox':
        return _from_jd_equinox(jd)

    else:
        return _from_jd_schematic(jd, method)