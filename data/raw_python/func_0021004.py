def getmany(self, *keys):
        """
        Return a list of values corresponding to the keys in the iterable of
        *keys*.
        If a key is not present in the collection, its corresponding value will
        be :obj:`None`.

        .. note::
            This method is not implemented by standard Python dictionary
            classes.
        """
        pickled_keys = (self._pickle_key(k) for k in keys)
        pickled_values = self.redis.hmget(self.key, *pickled_keys)

        ret = []
        for k, v in zip(keys, pickled_values):
            value = self.cache.get(k, self._unpickle(v))
            ret.append(value)

        return ret