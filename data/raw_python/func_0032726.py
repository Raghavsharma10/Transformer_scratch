def zcount(self, name, min, max):
        """
        Returns the number of elements in the sorted set at key ``name`` with
        a score between ``min`` and ``max``.

        :param name: str
        :param min: float
        :param max: float
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.zcount(self.redis_key(name), min, max)