def Fstar(self, value):
        """ set fixed effect design for predictions """
        if value is None:
            self._use_to_predict = False
        else:
            assert value.shape[1] == self._K, 'Dimension mismatch'
            self._use_to_predict = True
        self._Fstar = value
        self.clear_cache('predict')