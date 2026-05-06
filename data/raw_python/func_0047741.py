def add_die(self, die, count=1):
        '''Add ``Die`` to Roll.
        :param die: Die instance
        :param count: number of times die is rolled
        '''
        for x in range(count):
            self._dice.append(die)
        self._odds = None