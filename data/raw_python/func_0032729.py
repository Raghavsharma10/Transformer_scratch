def zremrangebyscore(self, name, min, max):
        """
        Remove a range of element by between score ``min_value`` and
        ``max_value`` both included.

        :param name: str     the name of the redis key
        :param min:
        :param max:
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.zremrangebyscore(self.redis_key(name), min, max)