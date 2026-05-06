def x_rolls(self, number, count=0, func=sum):
        '''Iterator of number dice rolls.
        :param count: [0] Return list of ``count`` sums
        :param func: [sum] Apply func to list of individual die rolls func([])
        '''
        for x in range(number):
            yield self.roll(count, func)