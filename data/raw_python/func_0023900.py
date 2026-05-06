def free_param_bounds(self):
        """Returns the bounds of the free hyperparameters.
        
        Returns
        -------
        free_param_bounds : :py:class:`Array`
            Array of the bounds of the free parameters, in order.
        """
        return scipy.concatenate((self.k1.free_param_bounds, self.k2.free_param_bounds))