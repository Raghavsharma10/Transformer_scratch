def paramsarray(self, value):
        """Set new `paramsarray` and update via `updateParams`."""
        nparams = len(self._index_to_param)
        assert (isinstance(value, scipy.ndarray) and value.ndim == 1), (
                "paramsarray must be 1-dim ndarray")
        assert len(value) == nparams, ("Assigning paramsarray to ndarray "
                "of the wrong length.")
        if (self._paramsarray is not None) and all(value == self._paramsarray):
            return # do not need to do anything if value has not changed
        # build `newvalues` to pass to `updateParams`
        newvalues = {}
        vectorized_params = {}
        for (i, param) in self._index_to_param.items():
            if isinstance(param, str):
                newvalues[param] = float(value[i])
            elif isinstance(param, tuple):
                (iparam, iparamindex) = param
                if iparam in vectorized_params:
                    assert iparamindex not in vectorized_params[iparam]
                    vectorized_params[iparam][iparamindex] = float(value[i])
                else:
                    vectorized_params[iparam] = {iparamindex:float(value[i])}
            else:
                raise ValueError("Invalid param type")
        for (param, paramd) in vectorized_params.items():
            assert set(paramd.keys()) == set(range(len(paramd)))
            newvalues[param] = scipy.array([paramd[i] for i in range(len(paramd))],
                    dtype='float')
        self.updateParams(newvalues)
        self._paramsarray = self.paramsarray