def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([p.rvs(size=size) for p in self.univariate_priors])