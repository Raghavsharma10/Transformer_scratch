def B(self,value):
        """ set phenotype """
        assert value.shape[0]==self._K, 'Dimension mismatch'
        assert value.shape[1]==1, 'Dimension mismatch'
        self._B = value
        self.clear_cache('predict','Yres')