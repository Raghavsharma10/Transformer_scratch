def popitem(self):
        """Remove and return an arbitrary ``(key, value)`` pair from
        the dictionary.

        :func:`popitem` is useful to destructively iterate over
        a dictionary, as often used in set algorithms. If
        the dictionary is empty, calling :func:`popitem` raises
        a :exc:`KeyError`.
        """
        def popitem_trans(pipe):
            try:
                pickled_key = pipe.hkeys(self.key)[0]
            except IndexError:
                raise KeyError

            # pop its value
            pipe.multi()
            pipe.hget(self.key, pickled_key)
            pipe.hdel(self.key, pickled_key)
            pickled_value, __ = pipe.execute()

            return (
                self._unpickle_key(pickled_key), self._unpickle(pickled_value)
            )

        key, value = self._transaction(popitem_trans)

        return key, self.cache.pop(key, value)