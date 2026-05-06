def pfcount(self, *sources):
        """
        Return the approximated cardinality of
        the set observed by the HyperLogLog at key(s).

        Using the execute_command because redis-py-cluster disabled it
        unnecessarily. but you can only send one key at a time in that case,
        or only keys that map to the same keyslot.
        Use at your own risk.

        :param sources: [str]     the names of the redis keys
        """
        sources = [self.redis_key(s) for s in sources]
        with self.pipe as pipe:
            return pipe.execute_command('PFCOUNT', *sources)