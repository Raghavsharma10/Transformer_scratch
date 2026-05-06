def timestamp_from_datetime(dt):
    """
    Compute timestamp from a datetime object that could be timezone aware
    or unaware.
    """
    try:
        utc_dt = dt.astimezone(pytz.utc)
    except ValueError:
        utc_dt = dt.replace(tzinfo=pytz.utc)
    return timegm(utc_dt.timetuple())