def retry_wait_time(attempt, cap):
    """
    Determine a retry wait time based on the number of the
    retry attempt and a cap on the wait time. The wait time
    uses an exponential backoff with a random jitter.
    The algorithm used is explained at
    https://www.awsarchitectureblog.com/2015/03/backoff.html.

    :param int attempt: The number of the attempt
    :param int cap: A cap on the wait time in milliseconds
    :returns: The number of milliseconds to wait
    :rtype: int
    """
    base = 100
    max_wait = min(cap, base * (2 ** attempt))
    return random.choice(range(0, max_wait))