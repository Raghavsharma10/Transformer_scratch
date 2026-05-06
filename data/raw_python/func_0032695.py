def substr(self, name, start, end=-1):
        """
        Return a substring of the string at key ``name``. ``start`` and ``end``
        are 0-based integers specifying the portion of the string to return.

        :param name: str     the name of the redis key
        :param start: int
        :param end: int
        :return: Future()
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.substr(self.redis_key(name), start=start, end=end)

            def cb():
                f.set(self.valueparse.decode(res.result))

            pipe.on_execute(cb)
            return f