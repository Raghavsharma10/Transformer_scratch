def hlen(self, name):
        """
        Returns the number of elements in the Hash.

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.hlen(self.redis_key(name))