def retry(retries=0, delay=timedelta(), conditions=[]):
    """
    A decorator for making a function that retries on failure.

    Args:
        retries (Integral): The number of times to retry if a failure occurs.
        delay (timedelta, optional, 0 seconds): A timedelta representing
            the amount of time to delay between retries.
        conditions (list): A list of retry conditions.
    """
    delay_in_seconds = delay.total_seconds()

    def decorator(function):
        """
        The actual decorator for retrying.
        """
        @wraps(function)
        def wrapper(*args, **kwargs):
            """
            The actual wrapper for retrying.
            """
            func = partial(function, *args, **kwargs)
            return retry_loop(retries, delay_in_seconds, conditions, func)

        return wrapper

    return decorator