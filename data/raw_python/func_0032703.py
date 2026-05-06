def sdiffstore(self, dest, *keys):
        """
        Store the difference of sets specified by ``keys`` into a new
        set named ``dest``.  Returns the number of keys in the new set.
        """
        keys = [self.redis_key(k) for k in self._parse_values(keys)]

        with self.pipe as pipe:
            return pipe.sdiffstore(self.redis_key(dest), *keys)