def every_minute(dt=datetime.datetime.utcnow(), fmt=None):
    """
    Just pass on the given date.
    """
    date = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 1, 0, dt.tzinfo)
    if fmt is not None:
        return date.strftime(fmt)
    return date