def zincrby(self, name, value, amount=1):
        """
        Increment the score of the item by `value`

        :param name: str     the name of the redis key
        :param value:
        :param amount:
        :return:
        """
        with self.pipe as pipe:
            return pipe.zincrby(self.redis_key(name),
                                value=self.valueparse.encode(value),
                                amount=amount)