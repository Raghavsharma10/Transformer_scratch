def do_run(self, count=1):
        '''Roll count dice, store results. Does all stats so might be slower
        than specific doFoo methods. But, it is proly faster than running
        each of those seperately to get same stats.

        Sets the following properties:
          - stats.bucket
          - stats.sum
          - stats.avr

        :param count: Number of rolls to make.
        '''
        if not self.roll.summable:
            raise Exception('Roll is not summable')
        h = dict()
        total = 0
        for roll in self.roll.x_rolls(count):
            total += roll
            h[roll] = h.get(roll, 0) + 1
        self._bucket = h
        self.sum = total
        self.avr = total / count