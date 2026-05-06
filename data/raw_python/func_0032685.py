def object(self, infotype, key):
        """
        get the key's info stats

        :param name: str     the name of the redis key
        :param subcommand: REFCOUNT | ENCODING | IDLETIME
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.object(infotype, self.redis_key(key))