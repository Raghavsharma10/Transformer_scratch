def retry_handler(retries=0, delay=timedelta(), conditions=[]):
    """
    A simple wrapper function that creates a handler function by using
    on the retry_loop function.

    Args:
        retries (Integral): The number of times to retry if a failure occurs.
        delay (timedelta, optional, 0 seconds): A timedelta representing
            the amount time to delay between retries.
        conditions (list): A list of retry conditions.
    Returns:
        function: The retry_loop function partialed.
    """
    delay_in_seconds = delay.total_seconds()
    return partial(retry_loop, retries, delay_in_seconds, conditions)