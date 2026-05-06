def incrby(self, name, amount=1):
        """
        increment the value for key by value: int

        :param name: str     the name of the redis key
        :param amount: int
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.incrby(self.redis_key(name), amount=amount)