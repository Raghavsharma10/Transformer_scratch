def dlogprior(self, param):
        """Equal to value of `basemodel.dlogprior`."""
        assert param in self.freeparams, "Invalid param: {0}".format(param)
        if param in self.distributionparams:
            return 0.0
        else:
            return self._models[0].dlogprior(param)