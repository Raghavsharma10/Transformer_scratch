def logliks(self, x):
        """Calculate log-likelihood of a feature x for each model

        Converts all values that are exactly 1 or exactly 0 to 0.999 and 0.001
        because they are out of range of the beta distribution.

        Parameters
        ----------
        x : numpy.array-like
            A single vector to estimate the log-likelihood of the models on

        Returns
        -------
        logliks : numpy.array
            Log-likelihood of these data in each member of the model's family
        """
        x = x.copy()

        # Replace exactly 0 and exactly 1 values with a very small number
        # (machine epsilon, the smallest number that this computer is capable
        # of storing) because 0 and 1 are not in the Beta distribution.
        x[x == 0] = VERY_SMALL_NUMBER
        x[x == 1] = 1 - VERY_SMALL_NUMBER

        return np.array([np.log(prob) + rv.logpdf(x[np.isfinite(x)]).sum()
                         for prob, rv in
                         zip(self.prob_parameters, self.rvs)])