def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([scipy.stats.norm.rvs(loc=m, scale=s, size=size) for s, m in zip(self.sigma, self.mu)])