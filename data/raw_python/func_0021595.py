def S2L(dd, mm, yy, timeZone=7):
    '''def S2L(dd, mm, yy, timeZone = 7): Convert solar date dd/mm/yyyy to
    the corresponding lunar date.'''
    dayNumber = jdFromDate(dd, mm, yy)
    k = int((dayNumber - 2415021.076998695) / 29.530588853)
    monthStart = getNewMoonDay(k + 1, timeZone)
    if (monthStart > dayNumber):
        monthStart = getNewMoonDay(k, timeZone)
    # alert(dayNumber + " -> " + monthStart)
    a11 = getLunarMonth11(yy, timeZone)
    b11 = a11
    if (a11 >= monthStart):
        lunarYear = yy
        a11 = getLunarMonth11(yy - 1, timeZone)
    else:
        lunarYear = yy + 1
        b11 = getLunarMonth11(yy + 1, timeZone)
    lunarDay = dayNumber - monthStart + 1
    diff = int((monthStart - a11) / 29.)

    lunarLeap = 0
    lunarMonth = diff + 11

    if (b11 - a11 > 365):
        leapMonthDiff = \
            getLeapMonthOffset(a11, timeZone)
        if (diff >= leapMonthDiff):
            lunarMonth = diff + 10
            if (diff == leapMonthDiff):
                lunarLeap = 1
    if (lunarMonth > 12):
        lunarMonth = lunarMonth - 12
    if (lunarMonth >= 11 and diff < 4):
        lunarYear -= 1
    # print [lunarDay, lunarMonth, lunarYear, lunarLeap]
    return \
        [lunarDay, lunarMonth, lunarYear, lunarLeap]