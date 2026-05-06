def update_coefs(self):
        """(Re)calculate the MA coefficients based on the instantaneous
        unit hydrograph."""
        coefs = []
        sum_coefs = 0.
        moment1 = self.iuh.moment1
        for t in itertools.count(0., 1.):
            points = (moment1 % 1,) if t <= moment1 <= (t+2.) else ()
            try:
                coef = integrate.quad(
                    self._quad, 0., 1., args=(t,), points=points)[0]
            except integrate.IntegrationWarning:
                idx = int(moment1)
                coefs = numpy.zeros(idx+2, dtype=float)
                weight = (moment1-idx)
                coefs[idx] = (1.-weight)
                coefs[idx+1] = weight
                self.coefs = coefs
                warnings.warn(
                    'During the determination of the MA coefficients '
                    'corresponding to the instantaneous unit hydrograph '
                    '`%s` a numerical integration problem occurred.  '
                    'Please check the calculated coefficients: %s.'
                    % (repr(self.iuh), objecttools.repr_values(coefs)))
                break   # pragma: no cover
            sum_coefs += coef
            if (sum_coefs > .9) and (coef < self.smallest_coeff):
                coefs = numpy.array(coefs)
                coefs /= numpy.sum(coefs)
                self.coefs = coefs
                break
            else:
                coefs.append(coef)