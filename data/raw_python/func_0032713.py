def lrange(self, name, start, stop):
        """
        Returns a range of items.

        :param name: str     the name of the redis key
        :param start: integer representing the start index of the range
        :param stop: integer representing the size of the list.
        :return: Future()
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.lrange(self.redis_key(name), start, stop)

            def cb():
                f.set([self.valueparse.decode(v) for v in res.result])

            pipe.on_execute(cb)
            return f