def persist(self, name):
        """
        clear any expiration TTL set on the object

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.persist(self.redis_key(name))