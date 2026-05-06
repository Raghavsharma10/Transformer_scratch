def srandmember(self, name, number=None):
        """
        Return a random member of the set.

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.srandmember(self.redis_key(name), number=number)

            def cb():
                if number is None:
                    f.set(self.valueparse.decode(res.result))
                else:
                    f.set([self.valueparse.decode(v) for v in res.result])

            pipe.on_execute(cb)
            return f