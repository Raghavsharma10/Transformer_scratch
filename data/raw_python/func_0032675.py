def dump(self, name):
        """
        get a redis RDB-like serialization of the object.

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.dump(self.redis_key(name))