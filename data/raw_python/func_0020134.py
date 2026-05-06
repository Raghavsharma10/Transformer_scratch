def pop_range(self, start, stop=None, withscores=True, **options):
        '''Remove and return a range from the ordered set by score.'''
        return self.backend.execute(
            self.client.zpopbyscore(self.id, start, stop,
                                    withscores=withscores, **options),
            partial(self._range, withscores))