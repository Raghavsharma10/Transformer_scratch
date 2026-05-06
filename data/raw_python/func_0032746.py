def hexists(self, name, key):
        """
        Returns ``True`` if the field exists, ``False`` otherwise.

        :param name: str     the name of the redis key
        :param key: the member of the hash
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.hexists(self.redis_key(name),
                                self.memberparse.encode(key))