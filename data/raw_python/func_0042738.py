def try_date(obj) -> Optional[datetime.date]:
    """Attempts to convert an object into a date.

    If the date format is known, it's recommended to use the corresponding function
    This is meant to be used in constructors.

    Parameters
    ----------
    obj: :class:`str`, :class:`datetime.datetime`, :class:`datetime.date`
        The object to convert.

    Returns
    -------
    :class:`datetime.date`, optional
        The represented date.
    """
    if obj is None:
        return None
    if isinstance(obj, datetime.datetime):
        return obj.date()
    if isinstance(obj, datetime.date):
        return obj
    res = parse_tibia_date(obj)
    if res is not None:
        return res
    res = parse_tibia_full_date(obj)
    if res is not None:
        return res
    res = parse_tibiadata_date(obj)
    return res