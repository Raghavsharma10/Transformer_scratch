def calc_all_ar_coefs(self, ar_order, ma_model):
        """Determine the AR coeffcients based on a least squares approach.

        The argument `ar_order` defines the number of AR coefficients to be
        determined.  The argument `ma_order` defines a pure |MA| model.
        The least squares approach is applied on all those coefficents of the
        pure MA model, which are associated with the part of the recession
        curve behind its turning point.

        The attribute |ARMA.rel_rmse| is updated with the resulting
        relative root mean square error.
        """
        turning_idx, _ = ma_model.turningpoint
        values = ma_model.coefs[turning_idx:]
        self.ar_coefs, residuals = numpy.linalg.lstsq(
            self.get_a(values, ar_order),
            self.get_b(values, ar_order),
            rcond=-1)[:2]
        if len(residuals) == 1:
            self._rel_rmse = numpy.sqrt(residuals[0])/numpy.sum(values)
        else:
            self._rel_rmse = 0.