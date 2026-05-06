def sismember(self, name, value):
        """
        Is the provided value is in the ``Set``?

        :param name: str     the name of the redis key
        :param value: str
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.sismember(self.redis_key(name),
                                  self.valueparse.encode(value))