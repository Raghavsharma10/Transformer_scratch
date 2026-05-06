def zrank(self, name, value):
        """
        Returns the rank of the element.

        :param name: str     the name of the redis key
        :param value: the element in the sorted set
        """
        with self.pipe as pipe:
            value = self.valueparse.encode(value)
            return pipe.zrank(self.redis_key(name), value)