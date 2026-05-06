def hget(self, name, key):
        """
        Returns the value stored in the field, None if the field doesn't exist.

        :param name: str     the name of the redis key
        :param key: the member of the hash
        :return: Future()
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.hget(self.redis_key(name),
                            self.memberparse.encode(key))

            def cb():
                f.set(self._value_decode(key, res.result))

            pipe.on_execute(cb)
            return f