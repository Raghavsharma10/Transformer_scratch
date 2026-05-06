def zlexcount(self, name, min, max):
        """
        Return the number of items in the sorted set between the
        lexicographical range ``min`` and ``max``.

        :param name: str     the name of the redis key
        :param min: int or '-inf'
        :param max: int or '+inf'
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.zlexcount(self.redis_key(name), min, max)