def L2S(lunarD, lunarM, lunarY, lunarLeap, tZ=7):
    '''def L2S(lunarD, lunarM, lunarY, lunarLeap, tZ = 7): Convert a lunar date
    to the corresponding solar date.'''
    if (lunarM < 11):
        a11 = getLunarMonth11(lunarY - 1, tZ)
        b11 = getLunarMonth11(lunarY, tZ)
    else:
        a11 = getLunarMonth11(lunarY, tZ)
        b11 = getLunarMonth11(lunarY + 1, tZ)
    k = int(0.5 +
            (a11 - 2415021.076998695) / 29.530588853)
    off = lunarM - 11
    if (off < 0):
        off += 12
    if (b11 - a11 > 365):
        leapOff = getLeapMonthOffset(a11, tZ)
        leapM = leapOff - 2
        if (leapM < 0):
            leapM += 12
        if (lunarLeap != 0 and lunarM != leapM):
            return [0, 0, 0]
        elif (lunarLeap != 0 or off >= leapOff):
            off += 1
    monthStart = getNewMoonDay(k + off, tZ)
    return jdToDate(monthStart + lunarD - 1)