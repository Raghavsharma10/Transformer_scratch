def front(self, *fields):
        '''Return the front pair of the structure'''
        v, f = tuple(self.irange(0, 0, fields=fields))
        if v:
            return (v[0], dict(((field, f[field][0]) for field in f)))