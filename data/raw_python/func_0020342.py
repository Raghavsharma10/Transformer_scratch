def zpopbyrank(self, name, start, stop=None, withscores=False, desc=False):
        '''Pop a range by rank.
        '''
        stop = stop if stop is not None else start
        return self.execute_script('zpop', (name,), 'rank', start,
                                   stop, int(desc), int(withscores),
                                   withscores=withscores)