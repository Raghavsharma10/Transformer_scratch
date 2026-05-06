def paramsarray(self):
        """All free model parameters as 1-dimensional `numpy.ndarray`.

        You are allowed to update model parameters by direct
        assignment of this property."""
        # Return copy of `_paramsarray` because setter checks if changed
        if self._paramsarray is not None:
            return self._paramsarray.copy()
        nparams = len(self._index_to_param)
        self._paramsarray = scipy.ndarray(shape=(nparams,), dtype='float')
        for (i, param) in self._index_to_param.items():
            if isinstance(param, str):
                self._paramsarray[i] = getattr(self.model, param)
            elif isinstance(param, tuple):
                self._paramsarray[i] = getattr(self.model, param[0])[param[1]]
            else:
                raise ValueError("Invalid param type")
        return self._paramsarray.copy()