def ts2dt(ts, milli=False, tz_aware=False):
    ''' convert timestamp int's (seconds) to datetime objects '''
    # anything already a datetime will still be returned
    # tz_aware, if set to true
    is_true(HAS_DATEUTIL, "`pip install python_dateutil` required")
    if isinstance(ts, (datetime, date)):
        pass
    elif is_empty(ts, except_=False):
        return None  # its not a timestamp
    elif isinstance(ts, (int, float, long)) and ts < 0:
        return None
    elif isinstance(ts, basestring):
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            # maybe we have a date like string already?
            try:
                ts = dt_parse(ts)
            except Exception:
                raise TypeError(
                    "unable to derive datetime from timestamp string: %s" % ts)
    elif milli:
        ts = float(ts) / 1000.  # convert milli to seconds
    else:
        ts = float(ts)  # already in seconds
    return _get_datetime(ts, tz_aware)