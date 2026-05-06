def Y(self, value):
        """ set phenotype """
        self._N = value.shape[0]
        self._P = value.shape[1]
        self._Y = value
        # missing data
        self._Iok = ~sp.isnan(value)
        self._veIok = vec(self._Iok)[:, 0]
        self._miss = (~self._Iok).any()
        # notify and clear_cached
        self.clear_cache('pheno')
        self._notify()
        self._notify('pheno')