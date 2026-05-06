def retry_loop(retries, delay_in_seconds, conditions, function):
    """
    Actually performs the retry loop used by the retry decorator
    and handler functions. Failures for retrying are defined by
    the RetryConditions passed in. If the maximum number of
    retries has been reached then it raises the most recent
    error or a ValueError on the most recent result value.

    Args:
        retries (Integral): Maximum number of times to retry.
        delay_in_seconds (Integral): Number of seconds to wait
            between retries.
        conditions (list): A list of retry conditions the can
            trigger a retry on a return value or exception.
        function (function): The function to wrap.

    Returns:
        value: The return value from function
    """
    if not isinstance(retries, Integral):
        raise TypeError(retries)

    if delay_in_seconds < 0:
        raise TypeError(delay_in_seconds)

    attempts = 0
    value = None
    err = None
    while attempts <= retries:
        try:
            value = function()
            for condition in conditions:
                if condition.on_value(value):
                    break
            else:
                return value
        except Exception as exc:
            err = exc
            for condition in conditions:
                if condition.on_exception(exc):
                    break
            else:
                raise

        attempts += 1
        sleep(delay_in_seconds)
    else:
        if err:
            raise err
        else:
            raise ValueError(
                "Max retries ({}) reached and return the value is still {}."
                .format(attempts, value)
            )

    return value