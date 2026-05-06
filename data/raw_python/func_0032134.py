def datetime_string(day, month, year, hour, minute):
    """Build a date string using the provided day, month, year numbers.

    Automatically adds a leading zero to ``day`` and ``month`` if they only have
    one digit.

    Args:
        day (int): Day number.
        month(int): Month number.
        year(int): Year number.
        hour (int): Hour of the day in 24h format.
        minute (int): Minute of the hour.

    Returns:
        str: Date in the format *YYYY-MM-DDThh:mm:ss*.
    """
    # Overflow
    if hour < 0 or hour > 23: hour = 0
    if minute < 0 or minute > 60: minute = 0

    return '%d-%02d-%02dT%02d:%02d:00' % (year, month, day, hour, minute)