def run_exitfuncs():
    """Function that behaves exactly like Python's atexit, but runs atexit functions
    in the order in which they were registered, not reversed.
    """
    exc_info = None
    for func, targs, kargs in _exithandlers:
        try:
            func(*targs, **kargs)
        except SystemExit:
            exc_info = sys.exc_info()
        except:
            exc_info = sys.exc_info()

    if exc_info is not None:
        six.reraise(exc_info[0], exc_info[1], exc_info[2])