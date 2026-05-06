def fire(self, args, kwargs):
        """
        Fire this signal with the specified arguments and keyword arguments.

        Typically this is used by using :meth:`__call__()` on this object which
        is more natural as it does all the argument packing/unpacking
        transparently.
        """
        for info in self._listeners[:]:
            if info.pass_signal:
                info.listener(*args, signal=self, **kwargs)
            else:
                info.listener(*args, **kwargs)