def ipop_range(self, start, stop=None, withscores=True, **options):
        '''Remove and return a range from the ordered set by rank (index).'''
        return self.backend.execute(
            self.client.zpopbyrank(self.id, start, stop,
                                   withscores=withscores, **options),
            partial(self._range, withscores))