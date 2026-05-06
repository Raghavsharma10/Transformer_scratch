def lrem(self, name, value, num=1):
        """
        Remove first occurrence of value.

        Can't use redis-py interface. It's inconstistent between
        redis.Redis and redis.StrictRedis in terms of the kwargs.
        Better to use the underlying execute_command instead.

        :param name: str     the name of the redis key
        :param num:
        :param value:
        :return: Future()
        """
        with self.pipe as pipe:
            value = self.valueparse.encode(value)
            return pipe.execute_command('LREM', self.redis_key(name),
                                        num, value)