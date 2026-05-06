def timeout(limit, handler):
    """A decorator ensuring that the decorated function tun time does not
    exceeds the argument limit.

    :args limit: the time limit
    :type limit: int

    :args handler: the handler function called when the decorated
    function times out.
    :type handler: callable

    Example:
    >>>def timeout_handler(limit, f, *args, **kwargs):
    ...     print "{func} call timed out after {lim}s.".format(
    ...         func=f.__name__, lim=limit)
    ...
    >>>@timeout(limit=5, handler=timeout_handler)
    ... def work(foo, bar, baz="spam")
    ...     time.sleep(10)
    >>>work("foo", "bar", "baz")
    # time passes...
    work call timed out after 5s.
    >>>


    """
    def wrapper(f):
        def wrapped_f(*args, **kwargs):
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(limit)
            try:
                res = f(*args, **kwargs)
            except Timeout:
                handler(limit, f, args, kwargs)
            else:
                return res
            finally:
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(0)
        return wrapped_f
    return wrapper