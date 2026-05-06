def residuals(self, pars, x, y, order):
        """
        Residual of Fourier Series.

        Parameters
        ----------
        pars : array_like
            Fourier series parameters.
        x : array_like
            An array of date.
        y : array_like
            An array of true values to fit.
        order : int
            An order of Fourier Series.
        """

        return y - self.fourier_series(pars, x, order)