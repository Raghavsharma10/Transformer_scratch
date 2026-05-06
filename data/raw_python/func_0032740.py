def hstrlen(self, name, key):
        """
        Return the number of bytes stored in the value of ``key``
        within hash ``name``
        """
        with self.pipe as pipe:
            return pipe.hstrlen(self.redis_key(name), key)