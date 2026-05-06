def expire(self, name, time):
        """
        Allow the key to expire after ``time`` seconds.

        :param name: str     the name of the redis key
        :param time: time expressed in seconds.
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.expire(self.redis_key(name), time)