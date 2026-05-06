def _spawn(func, *args, **kwargs):
    """
    Calls `func(*args, **kwargs)` in a daemon thread, and returns the (started)
    Thread object.
    """
    thr = Thread(target=func, args=args, kwargs=kwargs)
    thr.daemon = True
    thr.start()
    return thr