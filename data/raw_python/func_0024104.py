def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([scipy.stats.gamma.rvs(a, loc=0, scale=1.0 / b, size=size) for a, b in zip(self.a, self.b)])