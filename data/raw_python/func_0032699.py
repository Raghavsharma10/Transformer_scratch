def bitcount(self, name, start=None, end=None):
        """
        Returns the count of set bits in the value of ``key``.  Optional
        ``start`` and ``end`` paramaters indicate which bytes to consider

        :param name: str     the name of the redis key
        :param start: int
        :param end: int
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.bitcount(self.redis_key(name), start=start, end=end)