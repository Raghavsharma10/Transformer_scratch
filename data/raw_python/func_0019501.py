def create(cls, name, ncpus=None):
        """Create a Moap instance based on the predictor name.

        Parameters
        ----------
        name : str
            Name of the predictor (eg. Xgboost, BayesianRidge, ...)
        
        ncpus : int, optional
            Number of threads. Default is the number specified in the config.
        
        Returns
        -------
        moap : Moap instance
            moap instance.
        """
        try:
            return cls._predictors[name.lower()](ncpus=ncpus)
        except KeyError:
            raise Exception("Unknown class")