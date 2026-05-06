def perturbParams(self, pertSize=1e-3):
        """
        slightly perturbs the values of the parameters
        """
        params = self.getParams()
        self.setParams(params+pertSize*sp.randn(params.shape[0]))