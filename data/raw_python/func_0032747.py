def hincrby(self, name, key, amount=1):
        """
        Increment the value of the field.

        :param name: str     the name of the redis key
        :param increment: int
        :param field: str
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.hincrby(self.redis_key(name),
                                self.memberparse.encode(key),
                                amount)