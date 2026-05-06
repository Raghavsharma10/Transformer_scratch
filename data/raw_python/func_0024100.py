def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([scipy.stats.lognorm.rvs(s, loc=0, scale=em, size=size) for s, em in zip(self.sigma, self.emu)])