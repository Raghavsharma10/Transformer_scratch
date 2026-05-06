def zdiffstore(self, dest, keys, withscores=False):
        '''Compute the difference of multiple sorted.

        The difference of sets specified by ``keys`` into a new sorted set
        in ``dest``.
        '''
        keys = (dest,) + tuple(keys)
        wscores = 'withscores' if withscores else ''
        return self.execute_script('zdiffstore', keys, wscores,
                                   withscores=withscores)