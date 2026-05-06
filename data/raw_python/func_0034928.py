def Y(self,value):
        """ set phenotype """
        assert value.shape[1]==1, 'Dimension mismatch'
        self._N = value.shape[0]
        self._Y = value