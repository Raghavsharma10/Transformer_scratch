def lset(self, name, index, value):
        """
        Set the value in the list at index *idx*

        :param name: str     the name of the redis key
        :param value:
        :param index:
        :return: Future()
        """
        with self.pipe as pipe:
            value = self.valueparse.encode(value)
            return pipe.lset(self.redis_key(name), index, value)