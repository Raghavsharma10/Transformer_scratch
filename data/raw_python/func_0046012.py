def do_sum(self, count=1):
        '''Set self.sum, self.avr and return sum of dice rolled, count times.
        :param count: Number of rolls to make
        :return: Total sum of all rolls
        '''
        if not self.roll.summable:
            return 0
        self.sum = sum(self.roll.roll(count))
        self.avr = self.sum / count
        return self.sum