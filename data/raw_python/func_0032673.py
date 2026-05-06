def exists(self, name):
        """
        does the key exist in redis?

        :param name: str the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.exists(self.redis_key(name))