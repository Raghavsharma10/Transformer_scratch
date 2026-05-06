def hourly(dt=datetime.datetime.utcnow(), fmt=None):
    """
    Get a new datetime object every hour.
    """
    date = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, 1, 1, 0, dt.tzinfo)
    if fmt is not None:
        return date.strftime(fmt)
    return date