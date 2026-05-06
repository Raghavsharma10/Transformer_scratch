def ltrim(self, name, start, end):
        """
        Trim the list from start to end.

        :param name: str     the name of the redis key
        :param start:
        :param end:
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.ltrim(self.redis_key(name), start, end)