def to_datetime(jdc):
    '''Return a datetime for the input floating point Julian Day Count'''
    year, month, day = gregorian.from_jd(jdc)

    # in jdc: 0.0 = noon, 0.5 = midnight
    # the 0.5 changes it to 0.0 = midnight, 0.5 = noon
    frac = (jdc + 0.5) % 1

    hours = int(24 * frac)

    mfrac = frac * 24 - hours
    mins = int(60 * round(mfrac, 6))

    sfrac = mfrac * 60 - mins
    secs = int(60 * round(sfrac, 6))

    msfrac = sfrac * 60 - secs

    # down to ms, which are 1/1000 of a second
    ms = int(1000 * round(msfrac, 6))

    return datetime(year, month, day, int(hours), int(mins), int(secs), int(ms), tzinfo=utc)