def rank(self, dte):
        '''The rank of a given *dte* in the timeseries'''
        timestamp = self.pickler.dumps(dte)
        return self.backend_structure().rank(timestamp)