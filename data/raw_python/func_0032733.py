def zremrangebylex(self, name, min, max):
        """
        Remove all elements in the sorted set between the
        lexicographical range specified by ``min`` and ``max``.

        Returns the number of elements removed.
        :param name: str     the name of the redis key
        :param min: int or -inf
        :param max: into or +inf
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.zremrangebylex(self.redis_key(name), min, max)