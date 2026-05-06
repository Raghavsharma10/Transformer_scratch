def renamenx(self, src, dst):
        "Rename key ``src`` to ``dst`` if ``dst`` doesn't already exist"
        with self.pipe as pipe:
            return pipe.renamenx(self.redis_key(src), self.redis_key(dst))