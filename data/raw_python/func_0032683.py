def rename(self, src, dst):
        """
        Rename key ``src`` to ``dst``
        """
        with self.pipe as pipe:
            return pipe.rename(self.redis_key(src), self.redis_key(dst))