def zrangebyscore(self, name, min, max, start=None, num=None,
                      withscores=False, score_cast_func=float):
        """
        Returns the range of elements included between the scores (min and max)

        :param name: str     the name of the redis key
        :param min:
        :param max:
        :param start:
        :param num:
        :param withscores:
        :param score_cast_func:
        :return: Future()
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.zrangebyscore(self.redis_key(name), min, max,
                                     start=start, num=num,
                                     withscores=withscores,
                                     score_cast_func=score_cast_func)

            def cb():
                if withscores:
                    f.set([(self.valueparse.decode(v), s) for v, s in
                           res.result])
                else:
                    f.set([self.valueparse.decode(v) for v in res.result])

            pipe.on_execute(cb)
            return f