def bounds(self):
        """The bounds of the random variable.
        
        Set `self.i=0.95` to return the 95% interval if this is used for setting
        bounds on optimizers/etc. where infinite bounds may not be useful.
        """
        return [scipy.stats.lognorm.interval(self.i, s, loc=0, scale=em) for s, em in zip(self.sigma, self.emu)]