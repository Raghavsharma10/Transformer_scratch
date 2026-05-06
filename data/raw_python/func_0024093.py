def bounds(self):
        """The bounds of the random variable.
        
        Set `self.i=0.95` to return the 95% interval if this is used for setting
        bounds on optimizers/etc. where infinite bounds may not be useful.
        """
        return [scipy.stats.norm.interval(self.i, loc=m, scale=s) for s, m in zip(self.sigma, self.mu)]