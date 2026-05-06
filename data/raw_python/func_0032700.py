def incr(self, name, amount=1):
        """
        increment the value for key by 1

        :param name: str     the name of the redis key
        :param amount: int
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.incr(self.redis_key(name), amount=amount)