def hset(self, name, key, value):
        """
        Set ``member`` in the Hash at ``value``.

        :param name: str     the name of the redis key
        :param value:
        :param key: the member of the hash key
        :return: Future()
        """
        with self.pipe as pipe:
            value = self._value_encode(key, value)
            key = self.memberparse.encode(key)
            return pipe.hset(self.redis_key(name), key, value)