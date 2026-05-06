def solarEclipseGlobal(jd, backward):
    """ Returns the jd details of previous or next global solar eclipse. """

    sweList = swisseph.sol_eclipse_when_glob(jd, backward=backward)
    return {
        'maximum': sweList[1][0],
        'begin': sweList[1][2],
        'end': sweList[1][3],
        'totality_begin': sweList[1][4],
        'totality_end': sweList[1][5],
        'center_line_begin': sweList[1][6],
        'center_line_end': sweList[1][7],
    }