def setRandomParams(self):
        """
        set random hyperparameters
        """
        params = sp.randn(self.getNumberParams())
        self.setParams(params)