def append(self, name, value):
        """
        Appends the string ``value`` to the value at ``key``. If ``key``
        doesn't already exist, create it with a value of ``value``.
        Returns the new length of the value at ``key``.

        :param name: str     the name of the redis key
        :param value: str
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.append(self.redis_key(name),
                               self.valueparse.encode(value))