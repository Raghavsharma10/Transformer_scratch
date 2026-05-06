def setdefault(self, key, default=None):
        """If *key* is in the dictionary, return its value.
        If not, insert *key* with a value of *default* and
        return *default*. *default* defaults to :obj:`None`.
        """
        if key in self.cache:
            return self.cache[key]

        def setdefault_trans(pipe):
            pickled_key = self._pickle_key(key)

            pipe.multi()
            pipe.hsetnx(self.key, pickled_key, self._pickle_value(default))
            pipe.hget(self.key, pickled_key)

            __, pickled_value = pipe.execute()

            return self._unpickle(pickled_value)

        value = self._transaction(setdefault_trans)

        if self.writeback:
            self.cache[key] = value
        return value