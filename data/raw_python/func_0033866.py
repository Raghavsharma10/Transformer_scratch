def wtFromUTCpy(pyUTC, leapSecs=14):
    """convenience function:
         allows to use python UTC times and
         returns only week and tow"""
    ymdhms = ymdhmsFromPyUTC(pyUTC)
    wSowDSoD = apply(gpsFromUTC, ymdhms + (leapSecs,))
    return wSowDSoD[0:2]