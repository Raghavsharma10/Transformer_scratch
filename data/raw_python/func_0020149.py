def back(self, *fields):
        '''Return the back pair of the structure'''
        ts = self.irange(-1, -1, fields=fields)
        if ts:
            return ts.end(), ts[0]