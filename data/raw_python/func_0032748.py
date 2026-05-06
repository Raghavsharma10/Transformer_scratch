def hmget(self, name, keys, *args):
        """
        Returns the values stored in the fields.

        :param name: str     the name of the redis key
        :param fields:
        :return: Future()
        """
        member_encode = self.memberparse.encode
        keys = [k for k in self._parse_values(keys, args)]
        with self.pipe as pipe:
            f = Future()
            res = pipe.hmget(self.redis_key(name),
                             [member_encode(k) for k in keys])

            def cb():
                f.set([self._value_decode(keys[i], v)
                       for i, v in enumerate(res.result)])

            pipe.on_execute(cb)
            return f