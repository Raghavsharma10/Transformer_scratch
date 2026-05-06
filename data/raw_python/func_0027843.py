def get(self, key):
        """
        Get an entry from the cache by key.

        @raise KeyError: if the given key is not present in the cache.

        @raise CacheFault: (a L{KeyError} subclass) if the given key is present
            in the cache, but the value it points to is gone.
        """
        o = self.data[key]()
        if o is None:
            # On CPython, the weakref callback will always(?) run before any
            # other code has a chance to observe that the weakref is broken;
            # and since the callback removes the item from the dict, this
            # branch of code should never run. However, on PyPy (and possibly
            # other Python implementations), the weakref callback does not run
            # immediately, thus we may be able to observe this intermediate
            # state. Should this occur, we remove the dict item ourselves,
            # and raise CacheFault (which is a KeyError subclass).
            del self.data[key]
            raise CacheFault(
                "FinalizingCache has %r but its value is no more." % (key,))
        log.msg(interface=iaxiom.IStatEvent, stat_cache_hits=1, key=key)
        return o