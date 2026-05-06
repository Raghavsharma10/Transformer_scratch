def normalize_date(tmy_date, year):
    """change TMY3 date to an arbitrary year.

    Args:
        tmy_date (datetime): date to mangle.
        year (int): desired year.

    Returns:
        (None)
    """
    month = tmy_date.month
    day = tmy_date.day - 1
    hour = tmy_date.hour
    # hack to get around 24:00 notation
    if month is 1 and day is 0 and hour is 0:
        year = year + 1
    return datetime.datetime(year, month, 1) + \
        datetime.timedelta(days=day, hours=hour, minutes=0)