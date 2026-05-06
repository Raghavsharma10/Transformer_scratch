def hdel(self, name, *keys):
        """
        Delete one or more hash field.

        :param name: str     the name of the redis key
        :param keys: on or more members to remove from the key.
        :return: Future()
        """
        with self.pipe as pipe:
            m_encode = self.memberparse.encode
            keys = [m_encode(m) for m in self._parse_values(keys)]
            return pipe.hdel(self.redis_key(name), *keys)