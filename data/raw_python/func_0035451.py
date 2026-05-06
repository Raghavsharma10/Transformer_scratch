def setCovariance(self,cov):
        """ set hyperparameters from given covariance """
        chol = LA.cholesky(cov,lower=True)
        params = chol[sp.tril_indices(self.dim)]
        self.setParams(params)