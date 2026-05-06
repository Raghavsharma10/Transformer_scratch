def dt2ts(dt, drop_micro=False):
    ''' convert datetime objects to timestamp seconds (float) '''
    is_true(HAS_DATEUTIL, "`pip install python_dateutil` required")
    if is_empty(dt, except_=False):
        ts = None
    elif isinstance(dt, (int, long, float)):  # its a ts already
        ts = float(dt)
    elif isinstance(dt, basestring):  # convert to datetime first
        try:
            parsed_dt = float(dt)
        except (TypeError, ValueError):
            parsed_dt = dt_parse(dt)
        ts = dt2ts(parsed_dt)
    else:
        assert isinstance(dt, (datetime, date))
        # keep micros; see: http://stackoverflow.com/questions/7031031
        ts = ((
            timegm(dt.timetuple()) * 1000.0) +
            (dt.microsecond / 1000.0)) / 1000.0
    if ts is None:
        pass
    elif drop_micro:
        ts = float(int(ts))
    else:
        ts = float(ts)
    return ts