def mget(self, keys, *args):
        """
        Returns a list of values ordered identically to ``keys``
        """
        keys = [self.redis_key(k) for k in self._parse_values(keys, args)]
        with self.pipe as pipe:
            f = Future()
            res = pipe.mget(keys)

            def cb():
                decode = self.valueparse.decode
                f.set([None if r is None else decode(r) for r in res.result])

            pipe.on_execute(cb)
            return f