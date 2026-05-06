def iter(self, count=0, func=sum):
        '''Iterator of infinite dice rolls.
        :param count: [0] Return list of ``count`` sums
        :param func: [sum] Apply func to list of individual die rolls func([])
        '''
        while True:
            yield self.roll(count, func)