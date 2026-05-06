def hkeys(self, name):
        """
        Returns all fields name in the Hash.

        :param name: str the name of the redis key
        :return: Future
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.hkeys(self.redis_key(name))

            def cb():
                m_decode = self.memberparse.decode
                f.set([m_decode(v) for v in res.result])

            pipe.on_execute(cb)
            return f