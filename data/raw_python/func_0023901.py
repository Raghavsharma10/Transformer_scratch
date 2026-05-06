def free_param_names(self):
        """Returns the names of the free hyperparameters.
        
        Returns
        -------
        free_param_names : :py:class:`Array`
            Array of the names of the free parameters, in order.
        """
        return scipy.concatenate((self.k1.free_param_names, self.k2.free_param_names))