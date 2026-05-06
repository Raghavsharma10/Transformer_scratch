def _mask_cov_func(self, *args):
        """Masks the covariance function into a form usable by :py:func:`mpmath.diff`.
        
        Parameters
        ----------
        *args : `num_dim` * 2 floats
            The individual elements of Xi and Xj to be passed to :py:attr:`cov_func`.
        """
        # Have to do it in two cases to get the 1d unwrapped properly:
        if self.num_dim == 1:
            return self.cov_func(args[0], args[1], *self.params)
        else:
            return self.cov_func(args[:self.num_dim], args[self.num_dim:], *self.params)