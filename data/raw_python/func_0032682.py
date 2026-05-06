def pttl(self, name):
        """
        Returns the number of milliseconds until the key ``name`` will expire

        :param name: str    the name of the redis key
        :return:
        """
        with self.pipe as pipe:
            return pipe.pttl(self.redis_key(name))