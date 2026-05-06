def update(self):
        """Delete the coefficients of the pure MA model and also all MA and
        AR coefficients of the ARMA model.  Also calculate or delete the values
        of all secondary iuh parameters, depending on the completeness of the
        values of the primary parameters.
        """
        del self.ma.coefs
        del self.arma.ma_coefs
        del self.arma.ar_coefs
        if self.primary_parameters_complete:
            self.calc_secondary_parameters()
        else:
            for secpar in self._SECONDARY_PARAMETERS.values():
                secpar.__delete__(self)