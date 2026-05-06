def updateParams(self, newvalues, update_all=False):
        """See docs for `Model` abstract base class."""
        assert all(map(lambda x: x in self.freeparams, newvalues.keys())),\
                "Invalid entry in newvalues: {0}\nfreeparams: {1}".format(
                ', '.join(newvalues.keys()), ', '.join(self.freeparams))

        newvalues_list = [{} for k in range(self.ncats)]

        if update_all or any([param in self.distributionparams for param
                in newvalues.keys()]):
            self._d_distributionparams = {}
            for param in self.distributionparams:
                if param in newvalues:
                    _checkParam(param, newvalues[param], self.PARAMLIMITS,
                            self.PARAMTYPES)
                    setattr(self, param, copy.copy(newvalues[param]))
            self._lambdas = DiscreteGamma(self.alpha_lambda, self.beta_lambda,
                    self.ncats)
            for (k, l) in enumerate(self._lambdas):
                newvalues_list[k][self.distributedparam] = l
        for name in self.freeparams:
            if name not in self.distributionparams:
                if name in newvalues:
                    value = newvalues[name]
                    _checkParam(name, value, self.PARAMLIMITS, self.PARAMTYPES)
                    setattr(self, name, copy.copy(value))
                    for k in range(self.ncats):
                        newvalues_list[k][name] = value
                elif update_all:
                    for k in range(self.ncats):
                        newvalues_list[k][name] = getattr(self, name)

        assert len(newvalues_list) == len(self._models) == self.ncats
        for (k, newvalues_k) in enumerate(newvalues_list):
            self._models[k].updateParams(newvalues_k)

        # check to make sure all models have same parameter values
        for param in self.freeparams:
            if param not in self.distributionparams:
                pvalue = getattr(self, param)
                assert all([scipy.allclose(pvalue, getattr(model, param))
                        for model in self._models]), ("{0}\n{1}".format(
                        pvalue, '\n'.join([str(getattr(model, param))
                        for model in self._models])))