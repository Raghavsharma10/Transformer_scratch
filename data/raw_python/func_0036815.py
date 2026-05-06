def _parse_date(string: str) -> datetime.date:
    """Parse an ISO format date (YYYY-mm-dd).

    >>> _parse_date('1990-01-02')
    datetime.date(1990, 1, 2)
    """
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()