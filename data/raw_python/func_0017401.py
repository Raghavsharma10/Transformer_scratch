def sd(self):
        '''Calculate standard deviation of timeseries'''
        v = self.var()
        if len(v):
            return np.sqrt(v)
        else:
            return None