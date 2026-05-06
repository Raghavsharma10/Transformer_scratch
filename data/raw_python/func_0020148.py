def front(self, *fields):
        '''Return the front pair of the structure'''
        ts = self.irange(0, 0, fields=fields)
        if ts:
            return ts.start(), ts[0]