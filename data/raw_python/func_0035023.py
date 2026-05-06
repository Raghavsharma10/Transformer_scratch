def _paramlist_PartialLikelihoods(self):
        """List of parameters looped over in `_computePartialLikelihoods`."""
        if self._distributionmodel:
            return [param for param in self.model.freeparams +
                    [self.model.distributedparam] if param not in
                    self.model.distributionparams]
        else:
            return self.model.freeparams