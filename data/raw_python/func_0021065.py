def append(self, value):
        """Insert *value* at the end of this collection."""
        len_self = self.redis.rpush(self.key, self._pickle(value))

        if self.writeback:
            self.cache[len_self - 1] = value