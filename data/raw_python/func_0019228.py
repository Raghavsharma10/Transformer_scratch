def update_ma_coefs(self):
        """Determine the MA coefficients.

        The number of MA coefficients is subsequently increased until the
        required precision |ARMA.max_dev_coefs| is reached.  Otherwise,
        a |RuntimeError| is raised.
        """
        self.ma_coefs = []
        for ma_order in range(1, self.ma.order+1):
            self.calc_next_ma_coef(ma_order, self.ma)
            if self.dev_coefs < self.max_dev_coefs:
                self.norm_coefs()
                break
        else:
            with hydpy.pub.options.reprdigits(12):
                raise RuntimeError(
                    f'Method `update_ma_coefs` is not able to determine the '
                    f'MA coefficients of the ARMA model with the desired '
                    f'accuracy.  You can set the tolerance value '
                    f'´max_dev_coefs` to a higher value.  An accuracy of '
                    f'`{objecttools.repr_(self.dev_coefs)}` has been reached '
                    f'using `{self.ma.order}` MA coefficients.')
        if numpy.min(self.response) < 0.:
            warnings.warn(
                'Note that the smallest response to a standard impulse of the '
                'determined ARMA model is negative (`%s`).'
                % objecttools.repr_(numpy.min(self.response)))