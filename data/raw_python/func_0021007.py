def pop(self, key, default=__marker):
        """If *key* is in the dictionary, remove it and return its value,
        else return *default*. If *default* is not given and *key* is not
        in the dictionary, a :exc:`KeyError` is raised.
        """
        pickled_key = self._pickle_key(key)

        if key in self.cache:
            self.redis.hdel(self.key, pickled_key)
            return self.cache.pop(key)

        def pop_trans(pipe):
            pickled_value = pipe.hget(self.key, pickled_key)
            if pickled_value is None:
                if default is self.__marker:
                    raise KeyError(key)
                return default

            pipe.hdel(self.key, pickled_key)
            return self._unpickle(pickled_value)

        value = self._transaction(pop_trans)
        self.cache.pop(key, None)

        return value