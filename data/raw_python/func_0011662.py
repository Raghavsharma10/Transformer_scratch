def simple_wait(func):
    """
    Decorator for adding simple text wait animation to
    long running functions.

    Examples:
        >>> @animation.simple_wait
        >>> def long_running_function():
        >>>     ... 5 seconds later ...
        >>>     return
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        wait = Wait()
        wait.start()
        try:
            ret = func(*args, **kwargs)
        finally:
            wait.stop()
        sys.stdout.write('\n')
        return ret
    return wrapper