def blpop(self, keys, timeout=0):
        """
        LPOP a value off of the first non-empty list
        named in the ``keys`` list.

        If none of the lists in ``keys`` has a value to LPOP, then block
        for ``timeout`` seconds, or until a value gets pushed on to one
        of the lists.

        If timeout is 0, then block indefinitely.
        """
        map = {self.redis_key(k): k for k in self._parse_values(keys)}
        keys = map.keys()

        with self.pipe as pipe:
            f = Future()
            res = pipe.blpop(keys, timeout=timeout)

            def cb():
                if res.result:
                    k = map[res.result[0]]
                    v = self.valueparse.decode(res.result[1])

                    f.set((k, v))
                else:
                    f.set(res.result)

            pipe.on_execute(cb)
            return f