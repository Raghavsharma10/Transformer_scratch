def parse_datetime(v, default=None, tz=None, exceptions: bool=True) -> datetime:
    """
    Parses datetime
    :param v: Input string
    :param default: Default value if exceptions=False
    :param tz: Default pytz timezone or None if utc
    :param exceptions: Raise exception on error or not
    :return: datetime
    """
    try:
        t = dateutil_parse(v, default=datetime(2000, 1, 1))
        if tz is None:
            tz = pytz.utc
        return t if t.tzinfo else tz.localize(t)
    except Exception:
        if exceptions:
            raise ValidationError('Failed to parse datetime from "{}"'.format(v))
        return default