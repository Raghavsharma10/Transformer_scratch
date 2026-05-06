def _to_jd_schematic(year, month, day, method):
    '''Calculate JD using various leap-year calculation methods'''

    y0, y1, y2, y3, y4, y5 = 0, 0, 0, 0, 0, 0

    intercal_cycle_yrs, over_cycle_yrs, leap_suppression_yrs = None, None, None

    # Use the every-four-years method below year 16 (madler) or below 15 (romme)
    if ((method in (100, 'romme') and year < 15) or
            (method in (128, 'madler') and year < 17)):
        method = 4

    if method in (4, 'continuous'):
        # Leap years: 15, 19, 23, ...
        y5 = -365

    elif method in (100, 'romme'):
        year = year - 13
        y5 = DAYS_IN_YEAR * 12 + 3

        leap_suppression_yrs = 100.
        leap_suppression_days = 36524  # leap_cycle_days * 25 - 1

        intercal_cycle_yrs = 400.
        intercal_cycle_days = 146097  # leap_suppression_days * 4 + 1

        over_cycle_yrs = 4000.
        over_cycle_days = 1460969  # intercal_cycle_days * 10 - 1

    elif method in (128, 'madler'):
        year = year - 17
        y5 = DAYS_IN_YEAR * 16 + 4

        leap_suppression_days = 46751  # 32 * leap_cycle_days - 1
        leap_suppression_yrs = 128

    else:
        raise ValueError("Unknown leap year method. Try: continuous, romme, madler or equinox")

    if over_cycle_yrs:
        y0 = trunc(year / over_cycle_yrs) * over_cycle_days
        year = year % over_cycle_yrs

    # count intercalary cycles in days (400 years long or None)
    if intercal_cycle_yrs:
        y1 = trunc(year / intercal_cycle_yrs) * intercal_cycle_days
        year = year % intercal_cycle_yrs

    # count leap suppresion cycles in days (100 or 128 years long)
    if leap_suppression_yrs:
        y2 = trunc(year / leap_suppression_yrs) * leap_suppression_days
        year = year % leap_suppression_yrs

    y3 = trunc(year / LEAP_CYCLE_YEARS) * LEAP_CYCLE_DAYS
    year = year % LEAP_CYCLE_YEARS

    # Adjust 'year' by one to account for lack of year 0
    y4 = year * DAYS_IN_YEAR

    yj = y0 + y1 + y2 + y3 + y4 + y5

    mj = (month - 1) * 30

    return EPOCH + yj + mj + day - 1