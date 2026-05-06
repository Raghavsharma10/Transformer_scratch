def evaluate(self, inputs):
        """Evaluate the model.

        Parameters
        ----------
        inputs : number or ndarray
            Wavelengths in same unit as ``points``.

        Returns
        -------
        y : number or ndarray
            Flux or throughput in same unit as ``lookup_table``.

        """
        y = super(Empirical1D, self).evaluate(inputs)

        # Assume NaN at both ends need to be extrapolated based on
        # nearest end point.
        if self.fill_value is np.nan:
            # Cannot use sampleset() due to ExtinctionModel1D
            x = np.squeeze(self.points)

            if np.isscalar(y):  # pragma: no cover
                if inputs < x[0]:
                    y = self.lookup_table[0]
                elif inputs > x[-1]:
                    y = self.lookup_table[-1]
            else:
                y[inputs < x[0]] = self.lookup_table[0]
                y[inputs > x[-1]] = self.lookup_table[-1]

        return self._process_neg_flux(inputs, y)