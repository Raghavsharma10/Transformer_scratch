def sunionstore(self, dest, keys, *args):
        """
        Store the union of sets specified by ``keys`` into a new
        set named ``dest``.  Returns the number of members in the new set.
        """
        keys = [self.redis_key(k) for k in self._parse_values(keys, args)]
        with self.pipe as pipe:
            return pipe.sunionstore(self.redis_key(dest), *keys)