def setCovariance(self,cov):
        """ set hyperparameters from given covariance """
        self.setParams(sp.log(sp.diagonal(cov)))