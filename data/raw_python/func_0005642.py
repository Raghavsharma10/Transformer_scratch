def per_delta(start: datetime, end: datetime, delta: timedelta):
    """
    Iterates over time range in steps specified in delta.

    :param start: Start of time range (inclusive)
    :param end: End of time range (exclusive)
    :param delta: Step interval

    :return: Iterable collection of [(start+td*0, start+td*1), (start+td*1, start+td*2), ..., end)
    """
    curr = start
    while curr < end:
        curr_end = curr + delta
        yield curr, curr_end
        curr = curr_end