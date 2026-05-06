def _parse(yr, mo, day):
    """
    Basic parser to deal with date format of the Kp file.
    """
    
    yr = '20'+yr
    yr = int(yr)
    mo = int(mo)
    day = int(day)
    return pds.datetime(yr, mo, day)