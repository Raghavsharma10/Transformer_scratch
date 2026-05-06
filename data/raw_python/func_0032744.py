def hgetall(self, name):
        """
        Returns all the fields and values in the Hash.

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.hgetall(self.redis_key(name))

            def cb():
                data = {}
                m_decode = self.memberparse.decode
                v_decode = self._value_decode
                for k, v in res.result.items():
                    k = m_decode(k)
                    v = v_decode(k, v)
                    data[k] = v
                f.set(data)

            pipe.on_execute(cb)
            return f