def hscan(self, name, cursor=0, match=None, count=None):
        """
        Incrementally return key/value slices in a hash. Also return a cursor
        indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.hscan(self.redis_key(name), cursor=cursor,
                             match=match, count=count)

            def cb():
                data = {}
                m_decode = self.memberparse.decode
                for k, v in res[1].items():
                    k = m_decode(k)
                    v = self._value_decode(k, v)
                    data[k] = v

                f.set((res[0], data))

            pipe.on_execute(cb)
            return f