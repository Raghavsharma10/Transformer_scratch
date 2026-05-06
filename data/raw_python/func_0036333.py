def to_ts(date: datetime.date) -> float:
    """Convert date to timestamp.

    >>> to_ts(datetime.date(2001, 1, 2))
    978393600.0
    """
    return datetime.datetime(
        date.year, date.month, date.day,
        tzinfo=datetime.timezone.utc).timestamp()