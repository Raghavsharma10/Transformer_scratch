def set_hyperparams(self, new_params):
        """Sets the free hyperparameters to the new parameter values in new_params.
        
        Parameters
        ----------
        new_params : :py:class:`Array` or other Array-like, (len(:py:attr:`self.free_params`),)
            New parameter values, ordered as dictated by the docstring for the
            class.
        """
        new_params = scipy.asarray(new_params, dtype=float)
        
        if len(new_params) == len(self.free_params):
            if self.enforce_bounds:
                for idx, new_param, bound in zip(range(0, len(new_params)), new_params, self.free_param_bounds):
                    if bound[0] is not None and new_param < bound[0]:
                        new_params[idx] = bound[0]
                    elif bound[1] is not None and new_param > bound[1]:
                        new_params[idx] = bound[1]
            self.params[~self.fixed_params] = new_params
        else:
            raise ValueError("Length of new_params must be %s!" % (len(self.free_params),))