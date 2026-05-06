def zscan(self, name, cursor=0, match=None, count=None,
              score_cast_func=float):
        """
        Incrementally return lists of elements in a sorted set. Also return a
        cursor indicating the scan position.

        ``match`` allows for filtering the members by pattern

        ``count`` allows for hint the minimum number of returns

        ``score_cast_func`` a callable used to cast the score return value
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.zscan(self.redis_key(name), cursor=cursor,
                             match=match, count=count,
                             score_cast_func=score_cast_func)

            def cb():
                f.set((res[0], [(self.valueparse.decode(k), v)
                                for k, v in res[1]]))

            pipe.on_execute(cb)
            return f