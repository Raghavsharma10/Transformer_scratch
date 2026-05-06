def _sub_index_param(self, param):
        """Returns list of sub-indexes for `param`.

        Used in computing partial likelihoods; loop over these indices."""
        if self._distributionmodel and (param ==
                self.model.distributedparam):
            indices = [()]
        else:
            paramvalue = getattr(self.model, param)
            if isinstance(paramvalue, float):
                indices = [()]
            elif (isinstance(paramvalue, scipy.ndarray) and
                    paramvalue.ndim == 1 and paramvalue.shape[0] > 1):
                indices = [(j,) for j in range(len(paramvalue))]
            else:
                raise RuntimeError("Invalid param: {0}, {1}".format(
                        param, paramvalue))
        return indices