def compute_ll_matrix(self, bounds, num_pts):
        """Compute the log likelihood over the (free) parameter space.
        
        Parameters
        ----------
        bounds : 2-tuple or list of 2-tuples with length equal to the number of free parameters
            Bounds on the range to use for each of the parameters. If a single
            2-tuple is given, it will be used for each of the parameters.
        num_pts : int or list of ints with length equal to the number of free parameters
            If a single int is given, it will be used for each of the parameters.
        
        Returns
        -------
            ll_vals : :py:class:`Array`
                The log likelihood for each of the parameter possibilities.
            param_vals : List of :py:class:`Array`
                The parameter values used.
        """
        present_free_params = self.free_params[:]
        bounds = scipy.atleast_2d(scipy.asarray(bounds, dtype=float))
        if bounds.shape[1] != 2:
            raise ValueError("Argument bounds must have shape (n, 2)!")
        # If bounds is a single tuple, repeat it for each free parameter:
        if bounds.shape[0] == 1:
            bounds = scipy.tile(bounds, (len(present_free_params), 1))
        # If num_pts is a single value, use it for all of the parameters:
        try:
            iter(num_pts)
        except TypeError:
            num_pts = num_pts * scipy.ones(bounds.shape[0], dtype=int)
        else:
            num_pts = scipy.asarray(num_pts, dtype=int)
            if len(num_pts) != len(present_free_params):
                raise ValueError(
                    "Length of num_pts must match the number of free parameters!"
                )
        
        # Form arrays to evaluate parameters over:
        param_vals = []
        for k in xrange(0, len(present_free_params)):
            param_vals.append(scipy.linspace(bounds[k, 0], bounds[k, 1], num_pts[k]))
        ll_vals = self._compute_ll_matrix(0, param_vals, num_pts)
        
        # Reset the parameters to what they were before:
        self.update_hyperparameters(scipy.asarray(present_free_params, dtype=float))
        
        return (ll_vals, param_vals)