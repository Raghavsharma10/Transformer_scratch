def zrevrank(self, name, value):
        """
        Returns the ranking in reverse order for the member

        :param name: str     the name of the redis key
        :param member: str
        """
        with self.pipe as pipe:
            return pipe.zrevrank(self.redis_key(name),
                                 self.valueparse.encode(value))