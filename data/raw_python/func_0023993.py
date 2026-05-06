def _compute_ll_matrix(self, idx, param_vals, num_pts):
        """Recursive helper function for compute_ll_matrix.
        
        Parameters
        ----------
        idx : int
            The index of the parameter for this layer of the recursion to
            work on. `idx` == len(`num_pts`) is the base case that terminates
            the recursion.
        param_vals : List of :py:class:`Array`
            List of arrays of parameter values. Entries in the slots 0:`idx` are
            set to scalars by the previous levels of recursion.
        num_pts : :py:class:`Array`
            The numbers of points for each parameter.
        
        Returns
        -------
        vals : :py:class:`Array`
            The log likelihood for each of the parameter possibilities at lower
            levels.
        """
        if idx >= len(num_pts):
            # Base case: All entries in param_vals should be scalars:
            return -1.0 * self.update_hyperparameters(
                scipy.asarray(param_vals, dtype=float)
            )
        else:
            # Recursive case: call _compute_ll_matrix for each entry in param_vals[idx]:
            vals = scipy.zeros(num_pts[idx:], dtype=float)
            for k in xrange(0, len(param_vals[idx])):
                specific_param_vals = list(param_vals)
                specific_param_vals[idx] = param_vals[idx][k]
                vals[k] = self._compute_ll_matrix(
                    idx + 1,
                    specific_param_vals,
                    num_pts
                )
            return vals