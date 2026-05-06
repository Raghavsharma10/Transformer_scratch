def _postprocess(valuestr):
    """
    Takes value as str, returns str, int, float, date, datetime, or time
    """
    # USGS has started quoting time sometimes.  Grr, strip quotes in this case
    intpattern = re.compile(r'^\-?\d+$')
    floatpattern = re.compile(r'^\-?\d+\.\d+(E[+-]?\d\d+)?$')
    datedtpattern = '%Y-%m-%d'
    datedttimepattern = '%Y-%m-%dT%H:%M:%SZ'
    timedtpattern = '%H:%M:%S.%f'
    timepattern = re.compile(r'^\d{2}:\d{2}:\d{2}(\.\d{6})?')
    if valuestr.startswith('"') and valuestr.endswith('"'):
        # it's a string
        return valuestr[1:-1]
    elif re.match(intpattern, valuestr):
        # it's an integer
        return int(valuestr)
    elif re.match(floatpattern, valuestr):
        # floating point number
        return float(valuestr)
    # now let's try the datetime objects; throws exception if it doesn't match
    try:
        return datetime.datetime.strptime(valuestr, datedtpattern).date()
    except ValueError:
        pass
    try:
        return datetime.datetime.strptime(valuestr, datedttimepattern)
    except ValueError:
        pass
    # time parsing is complicated: Python's datetime module only accepts
    # fractions of a second only up to 6 digits
    mat = re.match(timepattern, valuestr)
    if mat:
        test = mat.group(0)
        try:
            return datetime.datetime.strptime(test, timedtpattern).time()
        except ValueError:
            pass

    # If we get here, we still haven't returned anything.
    logging.info(
        "The value %s couldn't be parsed as " % valuestr
        + "int, float, date, time, datetime. Returning it as string.")
    return valuestr