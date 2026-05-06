def zrange(self, name, start, end, desc=False, withscores=False,
               score_cast_func=float):
        """
        Returns all the elements including between ``start`` (non included)
        and ``stop`` (included).

        :param name: str     the name of the redis key
        :param start:
        :param end:
        :param desc:
        :param withscores:
        :param score_cast_func:
        :return:
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.zrange(
                self.redis_key(name), start, end, desc=desc,
                withscores=withscores, score_cast_func=score_cast_func)

            def cb():
                if withscores:
                    f.set([(self.valueparse.decode(v), s) for v, s in
                           res.result])
                else:
                    f.set([self.valueparse.decode(v) for v in res.result])

            pipe.on_execute(cb)
            return f