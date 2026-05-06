def setnx(self, name, value):
        """
        Set the value as a string in the key only if the key doesn't exist.

        :param name: str     the name of the redis key
        :param value:
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.setnx(self.redis_key(name),
                              self.valueparse.encode(value))