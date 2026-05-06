def updateParams(self, newvalues):
        """Update model parameters and re-compute likelihoods.

        This method is the **only** acceptable way to update model
        parameters. The likelihood is re-computed as needed
        by this method.

        Args:
            `newvalues` (dict)
                A dictionary keyed by param name and with value as new
                value to set. Each parameter name must either be a
                valid model parameter (in `model.freeparams`).
        """
        for (param, value) in newvalues.items():
            if param not in self.model.freeparams:
                raise RuntimeError("Can't handle param: {0}".format(
                        param))
        if newvalues:
            self.model.updateParams(newvalues)
            self._updateInternals()
            self._paramsarray = None