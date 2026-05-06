def bounds(self):
        """The bounds of the random variable.
        
        Set `self.i=0.95` to return the 95% interval if this is used for setting
        bounds on optimizers/etc. where infinite bounds may not be useful.
        """
        return [scipy.stats.gamma.interval(self.i, a, loc=0, scale=1.0 / b) for a, b in zip(self.a, self.b)]