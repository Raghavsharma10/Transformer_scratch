def retry(func=None, retries=5, backoff=None, exceptions=(IOError, OSError, EOFError), cleanup=None, sleep=time.sleep):
    """
    Decorator that retries the call ``retries`` times if ``func`` raises ``exceptions``. Can use a ``backoff`` function
    to sleep till next retry.

    Example::

        >>> should_fail = lambda foo=[1,2,3]: foo and foo.pop()
        >>> @retry
        ... def flaky_func():
        ...     if should_fail():
        ...         raise OSError('Tough luck!')
        ...     print("Success!")
        ...
        >>> flaky_func()
        Success!

    If it reaches the retry limit::

        >>> @retry
        ... def bad_func():
        ...     raise OSError('Tough luck!')
        ...
        >>> bad_func()
        Traceback (most recent call last):
        ...
        OSError: Tough luck!

    """

    @Aspect(bind=True)
    def retry_aspect(cutpoint, *args, **kwargs):
        for count in range(retries + 1):
            try:
                if count and cleanup:
                    cleanup(*args, **kwargs)
                yield
                break
            except exceptions as exc:
                if count == retries:
                    raise
                if not backoff:
                    timeout = 0
                elif isinstance(backoff, (int, float)):
                    timeout = backoff
                else:
                    timeout = backoff(count)
                logger.exception("%s(%s, %s) raised exception %s. %s retries left. Sleeping %s secs.",
                                 cutpoint.__name__, args, kwargs, exc, retries - count, timeout)
                sleep(timeout)

    return retry_aspect if func is None else retry_aspect(func)