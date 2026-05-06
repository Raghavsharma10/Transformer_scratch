def update_ar_coefs(self):
        """Determine the AR coefficients.

        The number of AR coefficients is subsequently increased until the
        required precision |ARMA.max_rel_rmse| is reached.  Otherwise,
        a |RuntimeError| is raised.
        """
        del self.ar_coefs
        for ar_order in range(1, self.effective_max_ar_order+1):
            self.calc_all_ar_coefs(ar_order, self.ma)
            if self._rel_rmse < self.max_rel_rmse:
                break
        else:
            with hydpy.pub.options.reprdigits(12):
                raise RuntimeError(
                    f'Method `update_ar_coefs` is not able to determine '
                    f'the AR coefficients of the ARMA model with the desired '
                    f'accuracy.  You can either set the tolerance value '
                    f'`max_rel_rmse` to a higher value or increase the '
                    f'allowed `max_ar_order`.  An accuracy of `'
                    f'{objecttools.repr_(self._rel_rmse)}` has been reached '
                    f'using `{self.effective_max_ar_order}` coefficients.')