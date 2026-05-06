def zscore(self, name, value):
        """
        Return the score of an element

        :param name: str     the name of the redis key
        :param value: the element in the sorted set key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.zscore(self.redis_key(name),
                               self.valueparse.encode(value))