def add_idle(self, callback, *args, **kwds):
    """Add an idle callback.

    An idle callback can return True, False or None.  These mean:

    - None: remove the callback (don't reschedule)
    - False: the callback did no work; reschedule later
    - True: the callback did some work; reschedule soon

    If the callback raises an exception, the traceback is logged and
    the callback is removed.
    """
    self.idlers.append((callback, args, kwds))