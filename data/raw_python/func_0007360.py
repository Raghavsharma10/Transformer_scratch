def now(timezone=None):
    """
    Return a naive datetime object for the given ``timezone``. A ``timezone``
    is any pytz- like or datetime.tzinfo-like timezone object. If no timezone
    is given, then UTC is assumed.

    This method is best used with pytz installed::

        pip install pytz
    """
    d = datetime.datetime.utcnow()
    if not timezone:
        return d

    return to_timezone(d, timezone).replace(tzinfo=None)