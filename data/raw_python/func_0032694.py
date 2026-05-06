def strlen(self, name):
        """
        Return the number of bytes stored in the value of the key

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.strlen(self.redis_key(name))