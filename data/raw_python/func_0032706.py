def scard(self, name):
        """
        How many items in the set?

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.scard(self.redis_key(name))