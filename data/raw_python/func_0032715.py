def lpop(self, name):
        """
        Pop the first object from the left.

        :param name: str     the name of the redis key
        :return: Future()

        """
        with self.pipe as pipe:
            f = Future()
            res = pipe.lpop(self.redis_key(name))

            def cb():
                f.set(self.valueparse.decode(res.result))

            pipe.on_execute(cb)
            return f