def x_rolls(self, number, count=0):
        '''Iterator of number dice rolls.
        :param count: [0] Return list of ``count`` sums
        '''
        for x in range(number):
            yield super(FuncRoll, self).roll(count, self._func)