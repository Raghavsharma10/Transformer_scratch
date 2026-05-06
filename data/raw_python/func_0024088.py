def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([numpy.random.uniform(low=b[0], high=b[1], size=size) for b in self.bounds])