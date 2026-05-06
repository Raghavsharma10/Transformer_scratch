def date_struct(year, month, day, tz = "UTC"):
    """
    Given year, month and day numeric values and a timezone
    convert to structured date object
    """
    ymdtz = (year, month, day, tz)
    if None in ymdtz:
        #logger.debug("a year, month, day or tz value was empty: %s" % str(ymdtz))
        return None # return early if we have a bad value
    try:
        return time.strptime("%s-%s-%s %s" % ymdtz,  "%Y-%m-%d %Z")
    except(TypeError, ValueError):
        #logger.debug("date failed to convert: %s" % str(ymdtz))
        pass