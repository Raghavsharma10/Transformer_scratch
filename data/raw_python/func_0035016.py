def dloglikarray(self):
        """Derivative of `loglik` with respect to `paramsarray`."""
        assert self.dparamscurrent, "dloglikarray requires paramscurrent == True"
        nparams = len(self._index_to_param)
        dloglikarray = scipy.ndarray(shape=(nparams,), dtype='float')
        for (i, param) in self._index_to_param.items():
            if isinstance(param, str):
                dloglikarray[i] = self.dloglik[param]
            elif isinstance(param, tuple):
                dloglikarray[i] = self.dloglik[param[0]][param[1]]
        return dloglikarray