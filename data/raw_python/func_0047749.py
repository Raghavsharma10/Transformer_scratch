def iter(self, count=0):
        '''Iterator of infinite dice rolls.
        :param count: [0] Return list of ``count`` sums
        '''
        while True:
            yield super(FuncRoll, self).roll(count, self._func)