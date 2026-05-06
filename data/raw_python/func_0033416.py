def get_timezone_converter(from_timezone, to_tz=None, tz_aware=False):
    '''
    return a function that converts a given
    datetime object from a timezone to utc

    :param from_timezone: timezone name as string
    '''
    if not from_timezone:
        return None
    is_true(HAS_DATEUTIL, "`pip install python_dateutil` required")
    is_true(HAS_PYTZ, "`pip install pytz` required")
    from_tz = pytz.timezone(from_timezone)
    return partial(_get_timezone_converter, from_tz=from_tz, to_tz=to_tz,
                   tz_aware=tz_aware)