def getbit(self, name, offset):
        """
        Returns a boolean indicating the value of ``offset`` in key

        :param name: str     the name of the redis key
        :param offset: int
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.getbit(self.redis_key(name), offset)