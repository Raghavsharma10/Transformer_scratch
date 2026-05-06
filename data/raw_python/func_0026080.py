def leap(year, method=None):
    '''
    Determine if this is a leap year in the FR calendar using one of three methods: 4, 100, 128
    (every 4th years, every 4th or 400th but not 100th, every 4th but not 128th)
    '''

    method = method or 'equinox'

    if year in (3, 7, 11):
        return True
    elif year < 15:
        return False

    if method in (4, 'continuous') or (year <= 16 and method in (128, 'madler', 4, 'continuous')):
        return year % 4 == 3

    elif method in (100, 'romme'):
        return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0

    elif method in (128, 'madler'):
        return year % 4 == 0 and year % 128 != 0

    elif method == 'equinox':
        # Is equinox on 366th day after (year, 1, 1)
        startjd = to_jd(year, 1, 1, method='equinox')
        if premier_da_la_annee(startjd + 367) - startjd == 366.0:
            return True
    else:
        raise ValueError("Unknown leap year method. Try: continuous, romme, madler or equinox")

    return False