def has_been(s, since, dt=None):
    '''
    A parser to check whether a (cron-like) string has been true during a certain time period.
    Useful for applications which cannot check every minute or need to catch up during a restart.
    @input:
        s = cron-like string (minute, hour, day of month, month, day of week)
        since = datetime to use as reference time for start of period
        dt = datetime to use as reference time for end of period, defaults to now
    @output: boolean of result
    '''
    if dt is None:
        dt = datetime.now(tz=since.tzinfo)

    if dt < since:
        raise ValueError("The since datetime must be before the current datetime.")

    while since <= dt:
        if is_now(s, since):
            return True
        since += timedelta(minutes=1)

    return False