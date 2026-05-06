def change(self, ident, rank=None):
        '''Change ident'''
        self.ident = ident
        if rank:
            self.rank = rank
            self.level = self._getLevel(rank, self.taxonomy)
        # count changes made to instance
        self.counter += 1