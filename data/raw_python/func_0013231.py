def is_now(s, dt=None):
    '''
    A very simple cron-like parser to determine, if (cron-like) string is valid for this date and time.
    @input:
        s = cron-like string (minute, hour, day of month, month, day of week)
        dt = datetime to use as reference time, defaults to now
    @output: boolean of result
    '''
    if dt is None:
        dt = datetime.now()
    minute, hour, dom, month, dow = s.split(' ')
    weekday = dt.isoweekday()

    return _parse_arg(minute, dt.minute) \
        and _parse_arg(hour, dt.hour) \
        and _parse_arg(dom, dt.day) \
        and _parse_arg(month, dt.month) \
        and _parse_arg(dow, 0 if weekday == 7 else weekday, True)