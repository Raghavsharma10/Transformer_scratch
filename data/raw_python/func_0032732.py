def zrevrangebylex(self, name, max, min, start=None, num=None):
        """
        Return the reversed lexicographical range of values from the sorted set
         between ``max`` and ``min``.

        If ``start`` and ``num`` are specified, then return a slice of the
        range.

        :param name: str     the name of the redis key
        :param max: int or '+inf'
        :param min: int or '-inf'
        :param start: int
        :param num: int
        :return: Future()
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.zrevrangebylex(self.redis_key(name), max, min,
                                      start=start, num=num)

            def cb():
                f.set([self.valueparse.decode(v) for v in res])

            pipe.on_execute(cb)
            return f