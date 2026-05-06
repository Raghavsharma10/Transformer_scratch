def start(milliseconds, func, *args, **kwargs):
    """
    Call function every interval.  Starts the timer at call time.
    Although this could also be a decorator, that would not initiate the time at
    the same time, so would require additional work.

    Arguments following function will be sent to function.  Note that these args
    are part of the defining state, and unless it is an object will reset each
    interval.

    The inine test will print "TickTock x.." every second, where x increments.

    >>> import time
    >>> class Tock(object):
    ...     count = 0
    ...     stop = None
    >>> def tick(obj):
    ...     obj.count += 1
    ...     if obj.stop and obj.count == 4:
    ...         obj.stop.set() # shut itself off
    ...         return
    ...     print("TickTock {}..".format(obj.count))
    >>> tock = Tock()
    >>> tock.stop = start(1000, tick, tock)
    >>> time.sleep(6)
    TickTock 1..
    TickTock 2..
    TickTock 3..
    """

    stopper = threading.Event()
    def interval(seconds, func, *args, **kwargs):
        """outer wrapper"""
        def wrapper():
            """inner wrapper"""
            if stopper.isSet():
                return
            interval(seconds, func, *args, **kwargs)
            try:
                func(*args, **kwargs)
            except: # pylint: disable=bare-except
                logging.error("Error during interval")
                logging.error(traceback.format_exc())

        thread = threading.Timer(seconds, wrapper)
        thread.daemon = True
        thread.start()
    interval(milliseconds/1000, func, *args, **kwargs)
    return stopper