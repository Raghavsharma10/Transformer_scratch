def set_hyperparams(self, new_params):
        """Set the (free) hyperparameters.
        
        Parameters
        ----------
        new_params : :py:class:`Array` or other Array-like
            New values of the free parameters.
        
        Raises
        ------
        ValueError
            If the length of `new_params` is not consistent with :py:attr:`self.params`.
        """
        new_params = scipy.asarray(new_params, dtype=float)
        
        if len(new_params) == len(self.free_params):
            num_free_k = sum(~self.k.fixed_params)
            self.k.set_hyperparams(new_params[:num_free_k])
            self.w.set_hyperparams(new_params[num_free_k:])
        else:
            raise ValueError("Length of new_params must be %s!" % (len(self.free_params),))