def rpoplpush(self, src, dst):
        """
        RPOP a value off of the ``src`` list and atomically LPUSH it
        on to the ``dst`` list.  Returns the value.
        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.rpoplpush(self.redis_key(src), self.redis_key(dst))

            def cb():
                f.set(self.valueparse.decode(res.result))

            pipe.on_execute(cb)
            return f