def SunLongitude(jdn):
    '''def SunLongitude(jdn): Compute the longitude of the sun at any time.
    Parameter: floating number jdn, the number of days since 1/1/4713 BC noon.
    '''
    T = (jdn - 2451545.0) / 36525.
    # Time in Julian centuries
    # from 2000-01-01 12:00:00 GMT
    T2 = T * T
    dr = math.pi / 180.  # degree to radian
    M = 357.52910 + 35999.05030 * T \
        - 0.0001559 * T2 - 0.00000048 * T * T2
    # mean anomaly, degree
    L0 = 280.46645 + 36000.76983 * T + 0.0003032 * T2
    # mean longitude, degree
    DL = (1.914600 - 0.004817 * T - 0.000014 * T2) \
        * math.sin(dr * M)
    DL += (0.019993 - 0.000101 * T) * math.sin(dr * 2 * M) \
        + 0.000290 * math.sin(dr * 3 * M)
    L = L0 + DL  # true longitude, degree
    L = L * dr
    L = L - math.pi * 2 * (float(L / (math.pi * 2)))
    # Normalize to (0, 2*math.pi)
    return L