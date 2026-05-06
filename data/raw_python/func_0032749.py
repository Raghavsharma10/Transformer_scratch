def hmset(self, name, mapping):
        """
        Sets or updates the fields with their corresponding values.

        :param name: str     the name of the redis key
        :param mapping: a dict with keys and values
        :return: Future()
        """
        with self.pipe as pipe:
            m_encode = self.memberparse.encode
            mapping = {m_encode(k): self._value_encode(k, v)
                       for k, v in mapping.items()}
            return pipe.hmset(self.redis_key(name), mapping)