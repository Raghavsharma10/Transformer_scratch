def calc_next_ma_coef(self, ma_order, ma_model):
        """Determine the MA coefficients of the ARMA model based on its
        predetermined AR coefficients and the MA ordinates of the given
        |MA| model.

        The MA coefficients are determined one at a time, beginning with the
        first one.  Each ARMA MA coefficient in set in a manner that allows
        for the exact reproduction of the equivalent pure MA coefficient with
        all relevant ARMA coefficients.
        """
        idx = ma_order-1
        coef = ma_model.coefs[idx]
        for jdx, ar_coef in enumerate(self.ar_coefs):
            zdx = idx-jdx-1
            if zdx >= 0:
                coef -= ar_coef*ma_model.coefs[zdx]
        self.ma_coefs = numpy.concatenate((self.ma_coefs, [coef]))