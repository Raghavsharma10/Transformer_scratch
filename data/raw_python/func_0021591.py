def NewMoon(k):
    '''def NewMoon(k): Compute the time of the k-th new moon after
    the new moon of 1/1/1900 13:52 UCT (measured as the number of
    days since 1/1/4713 BC noon UCT, e.g., 2451545.125 is 1/1/2000 15:00 UTC.
    Returns a floating number, e.g., 2415079.9758617813 for k=2 or
    2414961.935157746 for k=-2.'''
    # Time in Julian centuries from 1900 January 0.5
    T = k / 1236.85
    T2 = T * T
    T3 = T2 * T
    dr = math.pi / 180.
    Jd1 = 2415020.75933 + 29.53058868 * k \
        + 0.0001178 * T2 - 0.000000155 * T3
    Jd1 = Jd1 + 0.00033 * math.sin(
        (166.56 + 132.87 * T - 0.009173 * T2) * dr)
    # Mean new moon
    M = 359.2242 + 29.10535608 * k \
        - 0.0000333 * T2 - 0.00000347 * T3
    # Sun's mean anomaly
    Mpr = 306.0253 + 385.81691806 * k \
        + 0.0107306 * T2 + 0.00001236 * T3
    # Moon's mean anomaly
    F = 21.2964 + 390.67050646 * k - 0.0016528 * T2 \
        - 0.00000239 * T3
    # Moon's argument of latitude
    C1 = (0.1734 - 0.000393 * T) * math.sin(M * dr) \
        + 0.0021 * math.sin(2 * dr * M)
    C1 = C1 - 0.4068 * math.sin(Mpr * dr) \
        + 0.0161 * math.sin(dr * 2 * Mpr)
    C1 = C1 - 0.0004 * math.sin(dr * 3 * Mpr)
    C1 = C1 + 0.0104 * math.sin(dr * 2 * F) \
        - 0.0051 * math.sin(dr * (M + Mpr))
    C1 = C1 - 0.0074 * math.sin(dr * (M - Mpr)) \
        + 0.0004 * math.sin(dr * (2 * F + M))
    C1 = C1 - 0.0004 * math.sin(dr * (2 * F - M)) \
        - 0.0006 * math.sin(dr * (2 * F + Mpr))
    C1 = C1 + 0.0010 * math.sin(dr * (2 * F - Mpr)) \
        + 0.0005 * math.sin(dr * (2 * Mpr + M))
    if (T < -11):
        deltat = 0.001 + 0.000839 * T + 0.0002261 * T2 \
            - 0.00000845 * T3 - 0.000000081 * T * T3
    else:
        deltat = -0.000278 + 0.000265 * T + 0.000262 * T2
    JdNew = Jd1 + C1 - deltat
    return JdNew