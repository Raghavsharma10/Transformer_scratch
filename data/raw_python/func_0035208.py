def Lc(self,value):
        """ set col rotation """
        assert value.shape==(self.P, self.P), 'dimension mismatch'
        self._Lc = value
        self.clear_cache()