def keys(self, desc = None):
        '''numpy asarray does not copy data'''
        res = asarray(self.rc('index'))
        if desc == True:
            return reversed(res)
        else:
            return res