def to_timezone(dt, timezone):
    """
    Return an aware datetime which is ``dt`` converted to ``timezone``.

    If ``dt`` is naive, it is assumed to be UTC.

    For example, if ``dt`` is "06:00 UTC+0000" and ``timezone`` is "EDT-0400",
    then the result will be "02:00 EDT-0400".

    This method follows the guidelines in http://pytz.sourceforge.net/
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)
    return timezone.normalize(dt.astimezone(timezone))