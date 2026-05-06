def roll(self, count=0):
        '''Roll some dice!
        :param count: [0] Return list of sums
        :return: A single sum or list of ``count`` sums
        '''
        return super(FuncRoll, self).roll(count, self._func)