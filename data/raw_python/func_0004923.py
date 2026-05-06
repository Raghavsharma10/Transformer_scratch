def momentum(self, exponent=1, errorrequested=True):
        """Calculate momenta (integral of y times x^exponent)
        The integration is done by the trapezoid formula (np.trapz).

        Inputs:
            exponent: the exponent of q in the integration.
            errorrequested: True if error should be returned (true Gaussian
                error-propagation of the trapezoid formula)
        """
        y = self.Intensity * self.q ** exponent
        m = np.trapz(y, self.q)
        if errorrequested:
            err = self.Error * self.q ** exponent
            dm = errtrapz(self.q, err)
            return ErrorValue(m, dm)
        else:
            return m