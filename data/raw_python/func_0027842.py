def uncache(self, key, value):
        """
        Remove a key from the cache.

        As a sanity check, if the specified key is present in the cache, it
        must have the given value.

        @param key: The key to remove.

        @param value: The expected value for the key.
        """
        try:
            assert self.get(key) is value
            del self.data[key]
        except KeyError:
            # If the entry has already been removed from the cache, this will
            # result in KeyError which we ignore. If the entry is still in the
            # cache, but the weakref has been broken, this will result in
            # CacheFault (a KeyError subclass) which we also ignore. See the
            # comment in get() for an explanation of why this might happen.
            pass