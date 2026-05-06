def W(self,value):
        """ set fixed effect design """
        if value is None:   value = sp.zeros((self._N, 0))
        assert value.shape[0]==self._N, 'Dimension mismatch'
        self._K = value.shape[1]
        self._W = value
        self._notify()
        self.clear_cache('predict_in_sample','Yres')