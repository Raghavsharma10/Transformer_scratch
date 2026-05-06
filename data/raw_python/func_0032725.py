def zcard(self, name):
        """
        Returns the cardinality of the SortedSet.

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.zcard(self.redis_key(name))