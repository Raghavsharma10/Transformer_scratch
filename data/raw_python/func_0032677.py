def restore(self, name, value, pttl=0):
        """
        Restore serialized dump of a key back into redis

        :param name: the name of the key
        :param value: the binary representation of the key.
        :param pttl: milliseconds till key expires
        :return:
        """
        with self.pipe as pipe:
            res = pipe.restore(self.redis_key(name), ttl=pttl, value=value)
            f = Future()

            def cb():
                f.set(self.valueparse.decode(res.result))

            pipe.on_execute(cb)
            return f