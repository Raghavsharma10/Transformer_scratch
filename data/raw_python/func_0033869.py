def GpsSecondsFromPyUTC( pyUTC, leapSecs=14 ):
    """converts the python epoch to gps seconds

    pyEpoch = the python epoch from time.time()
    """
    t = t=gpsFromUTC(*ymdhmsFromPyUTC( pyUTC ))
    return int(t[0] * 60 * 60 * 24 * 7 + t[1])