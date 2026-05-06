def llen(self, name):
        """
        Returns the length of the list.

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.llen(self.redis_key(name))