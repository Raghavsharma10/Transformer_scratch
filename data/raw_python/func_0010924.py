def set_params(self, targets=None):
        '''Set the values of the parameters to the given target values.

        Parameters
        ----------
        targets : sequence of ndarray, optional
            Arrays for setting the parameters of our model. If this is not
            provided, the current best parameters for this optimizer will be
            used.
        '''
        if not isinstance(targets, (list, tuple)):
            targets = self._best_params
        for param, target in zip(self._params, targets):
            param.set_value(target)