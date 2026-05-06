def scan(self, cursor=0, match=None, count=None):
        """
        Incrementally return lists of key names. Also return a cursor
        indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
        f = Future()
        if self.keyspace is None:
            with self.pipe as pipe:
                res = pipe.scan(cursor=cursor, match=match, count=count)

                def cb():
                    f.set((res[0], [self.keyparse.decode(v) for v in res[1]]))

                pipe.on_execute(cb)
                return f

        if match is None:
            match = '*'
        match = "%s{%s}" % (self.keyspace, match)
        pattern = re.compile(r'^%s\{(.*)\}$' % self.keyspace)

        with self.pipe as pipe:

            res = pipe.scan(cursor=cursor, match=match, count=count)

            def cb():
                keys = []
                for k in res[1]:
                    k = self.keyparse.decode(k)
                    m = pattern.match(k)
                    if m:
                        keys.append(m.group(1))

                f.set((res[0], keys))

            pipe.on_execute(cb)
            return f