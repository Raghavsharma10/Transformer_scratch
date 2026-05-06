def _convert_timedelta_to_seconds(timedelta):
    """Returns the total seconds calculated from the supplied timedelta.

       (Function provided to enable running on Python 2.6 which lacks timedelta.total_seconds()).
    """

    days_in_seconds = timedelta.days * 24 * 3600
    return int((timedelta.microseconds + (timedelta.seconds + days_in_seconds) * 10 ** 6) / 10 ** 6)