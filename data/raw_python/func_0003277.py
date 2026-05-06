def date_time_string(timestamp=None):
    """Return the current date and time formatted for a message header."""
    global _last_date_time_string
    _last_timestamp, _last_str = _last_date_time_string
    if timestamp is None:
        timestamp = time.time()
    _curr_timestamp = int(timestamp)
    if _curr_timestamp == _last_timestamp:
        return _last_str
    else:
        year, month, day, hh, mm, ss, wd, y, z = time.gmtime(timestamp)
        s = b"%s, %02d %3s %4d %02d:%02d:%02d GMT" % (
                weekdayname[wd],
                day, monthname[month], year,
                hh, mm, ss)
        _last_date_time_string = (_curr_timestamp, s)
        return s