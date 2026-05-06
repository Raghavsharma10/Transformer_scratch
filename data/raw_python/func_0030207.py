def pages_count(self):
        '''Number of pages.'''
        if not self.limit or self.count<self.limit:
            return 1
        if self.count % self.limit <= self.orphans:
            return self.count // self.limit
        return int(math.ceil(float(self.count)/self.limit))