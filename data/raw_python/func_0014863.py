def random_date(dt_from, dt_to):
    """
    This function will return a random datetime between two datetime objects.
    :param start:
    :param end:
    """
    delta = dt_to - dt_from
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return dt_from + datetime.timedelta(seconds=random_second)