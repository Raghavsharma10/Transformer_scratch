def try_datetime(obj) -> Optional[datetime.datetime]:
    """Attempts to convert an object into a datetime.

    If the date format is known, it's recommended to use the corresponding function
    This is meant to be used in constructors.

    Parameters
    ----------
    obj: :class:`str`, :class:`dict`, :class:`datetime.datetime`
        The object to convert.

    Returns
    -------
    :class:`datetime.datetime`, optional
        The represented datetime, or ``None`` if conversion wasn't possible.
    """
    if obj is None:
        return None
    if isinstance(obj, datetime.datetime):
        return obj
    res = parse_tibia_datetime(obj)
    if res is not None:
        return res
    res = parse_tibiadata_datetime(obj)
    return res