def transform(self, X, y=None, **params):
        """
        Transforms *X* from phase-space to Fourier-space, returning the design
        matrix produced by :func:`Fourier.design_matrix` for input to a
        regressor.

        **Parameters**

        X : array-like, shape = [n_samples, 1]
            Column vector of phases.
        y : None, optional
            Unused argument for conformity (default None).

        **Returns**

        design_matrix : array-like, shape = [n_samples, 2*degree+1]
            Fourier design matrix produced by :func:`Fourier.design_matrix`.
        """
        data = numpy.dstack((numpy.array(X).T[0], range(len(X))))[0]
        phase, order = data[data[:,0].argsort()].T
        design_matrix = self.design_matrix(phase, self.degree)
        return design_matrix[order.argsort()]