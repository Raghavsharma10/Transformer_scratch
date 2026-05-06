def zremrangebyrank(self, name, min, max):
        """
        Remove a range of element between the rank ``start`` and
        ``stop`` both included.

        :param name: str     the name of the redis key
        :param min:
        :param max:
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.zremrangebyrank(self.redis_key(name), min, max)