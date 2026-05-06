def delete(self, *names):
        """
        Remove the key from redis

        :param names: tuple of strings - The keys to remove from redis.
        :return: Future()
        """
        names = [self.redis_key(n) for n in names]
        with self.pipe as pipe:
            return pipe.delete(*names)