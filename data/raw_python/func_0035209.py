def d(self,value):
        """ set anisotropic scaling """
        assert value.shape[0]==self.P*self.N, 'd dimension mismatch'
        self._d = value
        self.clear_cache()