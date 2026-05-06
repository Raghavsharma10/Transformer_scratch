def dlogprior(self, param):
        """Value of derivative of prior depends on value of `prior`."""
        assert param in self.freeparams, "Invalid param: {0}".format(param)
        return self._dlogprior[param]