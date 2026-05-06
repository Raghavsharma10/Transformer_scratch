def _substitute_fixed_parameters_covar(self, covar):
        """Insert fixed parameters in a covariance matrix"""
        covar_resolved = np.empty((len(self._fixed_parameters), len(self._fixed_parameters)))
        indices_of_fixed_parameters = [i for i in range(len(self.parameters())) if
                                       self._fixed_parameters[i] is not None]
        indices_of_free_parameters = [i for i in range(len(self.parameters())) if self._fixed_parameters[i] is None]
        for i in range(covar_resolved.shape[0]):
            if i in indices_of_fixed_parameters:
                # the i-eth argument was fixed. This means that the row and column corresponding to this argument
                # must be None
                covar_resolved[i, :] = 0
                continue
            for j in range(covar_resolved.shape[1]):
                if j in indices_of_fixed_parameters:
                    covar_resolved[:, j] = 0
                    continue
                covar_resolved[i, j] = covar[indices_of_free_parameters.index(i), indices_of_free_parameters.index(j)]
        return covar_resolved