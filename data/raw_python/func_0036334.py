def to_date(ts: float) -> datetime.date:
    """Convert timestamp to date.

    >>> to_date(978393600.0)
    datetime.date(2001, 1, 2)
    """
    return datetime.datetime.fromtimestamp(
        ts, tz=datetime.timezone.utc).date()