def values(self, desc = None):
        '''numpy asarray does not copy data'''
        if self._ts:
            res = asarray(self._ts)
            if desc == True:
                return reversed(res)
            else:
                return res
        else:
            return ndarray([0,0])