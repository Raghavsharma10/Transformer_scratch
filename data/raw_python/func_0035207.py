def Lr(self,value):
        """ set row rotation """
        assert value.shape==(self.N, self.N), 'dimension mismatch'
        self._Lr = value
        self.clear_cache()