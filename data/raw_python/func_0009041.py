def convert_timestamp(obj):
    """Make an ARF timestamp from an object.

    Argument can be a datetime.datetime object, a time.struct_time, an integer,
    a float, or a tuple of integers. The returned value is a numpy array with
    the integer number of seconds since the Epoch and any additional
    microseconds.

    Note that because floating point values are approximate, the conversion
    between float and integer tuple may not be reversible.

    """
    import numbers
    from datetime import datetime
    from time import mktime, struct_time
    from numpy import zeros

    out = zeros(2, dtype='int64')
    if isinstance(obj, datetime):
        out[0] = mktime(obj.timetuple())
        out[1] = obj.microsecond
    elif isinstance(obj, struct_time):
        out[0] = mktime(obj)
    elif isinstance(obj, numbers.Integral):
        out[0] = obj
    elif isinstance(obj, numbers.Real):
        out[0] = obj
        out[1] = (obj - out[0]) * 1e6
    else:
        try:
            out[:2] = obj[:2]
        except:
            raise TypeError("unable to convert %s to timestamp" % obj)
    return out