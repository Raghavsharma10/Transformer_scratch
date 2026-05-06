def sscan(self, name, cursor=0, match=None, count=None):
        """
        Incrementally return lists of elements in a set. Also return a cursor
        indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        :param name: str     the name of the redis key
        :param cursor: int
        :param match: str
        :param count: int
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.sscan(self.redis_key(name), cursor=cursor,
                             match=match, count=count)

            def cb():
                f.set((res[0], [self.valueparse.decode(v) for v in res[1]]))

            pipe.on_execute(cb)
            return f