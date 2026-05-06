def _from_jd_schematic(jd, method):
    '''Convert from JD using various leap-year calculation methods'''
    if jd < EPOCH:
        raise ValueError("Can't convert days before the French Revolution")

    # days since Epoch
    J = trunc(jd) + 0.5 - EPOCH

    y0, y1, y2, y3, y4, y5 = 0, 0, 0, 0, 0, 0
    intercal_cycle_days = leap_suppression_days = over_cycle_days = None

    # Use the every-four-years method below year 17
    if (J <= DAYS_IN_YEAR * 12 + 3 and
            method in (100, 'romme')) or (J <= DAYS_IN_YEAR * 17 + 4 and method in (128, 'madler')):
        method = 4

    # set p and r in Hatcher algorithm
    if method in (4, 'continuous'):
        # Leap years: 15, 19, 23, ...
        # Reorganize so that leap day is last day of cycle
        J = J + 365
        y5 = - 1

    elif method in (100, 'romme'):
        # Year 15 is not a leap year
        # Year 16 is leap, then multiples of 4, not multiples of 100, yes multiples of 400
        y5 = 12
        J = J - DAYS_IN_YEAR * 12 - 3

        leap_suppression_yrs = 100.
        leap_suppression_days = 36524  # LEAP_CYCLE_DAYS * 25 - 1

        intercal_cycle_yrs = 400.
        intercal_cycle_days = 146097  # leap_suppression_days * 4 + 1

        over_cycle_yrs = 4000.
        over_cycle_days = 1460969  # intercal_cycle_days * 10 - 1

    elif method in (128, 'madler'):
        # Year 15 is a leap year, then year 20 and multiples of 4, not multiples of 128
        y5 = 16
        J = J - DAYS_IN_YEAR * 16 - 4

        leap_suppression_yrs = 128
        leap_suppression_days = 46751  # 32 * leap_cycle_days - 1

    else:
        raise ValueError("Unknown leap year method. Try: continuous, romme, madler or equinox")

    if over_cycle_days:
        y0 = trunc(J / over_cycle_days) * over_cycle_yrs
        J = J % over_cycle_days

    if intercal_cycle_days:
        y1 = trunc(J / intercal_cycle_days) * intercal_cycle_yrs
        J = J % intercal_cycle_days

    if leap_suppression_days:
        y2 = trunc(J / leap_suppression_days) * leap_suppression_yrs
        J = J % leap_suppression_days

    y3 = trunc(J / LEAP_CYCLE_DAYS) * LEAP_CYCLE_YEARS

    if J % LEAP_CYCLE_DAYS == LEAP_CYCLE_DAYS - 1:
        J = 1460
    else:
        J = J % LEAP_CYCLE_DAYS

    # 0 <= J <= 1460
    # J needs to be 365 here on leap days ONLY

    y4 = trunc(J / DAYS_IN_YEAR)

    if J == DAYS_IN_YEAR * 4:
        y4 = y4 - 1
        J = 365.0
    else:
        J = J % DAYS_IN_YEAR

    year = y0 + y1 + y2 + y3 + y4 + y5

    month = trunc(J / 30.)
    J = J - month * 30

    return year + 1, month + 1, trunc(J) + 1