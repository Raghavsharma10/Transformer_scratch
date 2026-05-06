def F(self,value):
        """ set phenotype """
        assert value.shape[0]==self._N, 'Dimension mismatch'
        self._K = value.shape[1]
        self._F = value
        self.clear_cache('predict','Yres')