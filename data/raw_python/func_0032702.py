def incrbyfloat(self, name, amount=1.0):
        """
        increment the value for key by value: float

        :param name: str     the name of the redis key
        :param amount: int
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.incrbyfloat(self.redis_key(name), amount=amount)