def do_bucket(self, count=1):
        '''Set self.bucket and return results.
        :param count: Number of rolls to make
        :return: List of tuples (total of roll, times it was rolled)
        '''
        self._bucket = dict()
        for roll in self.roll.roll(count):
            self._bucket[roll] = self._bucket.get(roll, 0) + 1
        return self.bucket