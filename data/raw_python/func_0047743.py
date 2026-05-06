def roll(self, count=0, func=sum):
        '''Roll some dice!
        :param count: [0] Return list of sums
        :param func: [sum] Apply func to list of individual die rolls func([])
        :return: A single sum or list of ``count`` sums
        '''
        if count:
            return [func([die.roll() for die in self._dice]) for x in range(0, count)]
        else:
            return func([die.roll() for die in self._dice])