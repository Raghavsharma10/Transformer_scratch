def ttl(self, name):
        """
        get the number of seconds until the key's expiration

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.ttl(self.redis_key(name))