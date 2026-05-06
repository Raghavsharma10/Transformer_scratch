def per_month(start: datetime, end: datetime, n: int=1):
    """
    Iterates over time range in one month steps.
    Clamps to number of days in given month.

    :param start: Start of time range (inclusive)
    :param end: End of time range (exclusive)
    :param n: Number of months to step. Default is 1.

    :return: Iterable collection of [(month+0, month+1), (month+1, month+2), ..., end)
    """
    curr = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while curr < end:
        curr_end = add_month(curr, n)
        yield curr, curr_end
        curr = curr_end