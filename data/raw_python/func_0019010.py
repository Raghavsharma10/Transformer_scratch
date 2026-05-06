def calculate_error(self):
        """Estimate the numerical error based on the fluxes calculated
        by the current and the last method.

        >>> from hydpy.models.test_v1 import *
        >>> parameterstep()
        >>> model.numvars.idx_method = 2
        >>> results = numpy.asarray(fluxes.fastaccess._q_results)
        >>> results[:4] = 0., 3., 4., 0.
        >>> model.calculate_error()
        >>> from hydpy import round_
        >>> round_(model.numvars.error)
        1.0
        """
        self.numvars.error = 0.
        fluxes = self.sequences.fluxes
        for flux in fluxes.numerics:
            results = getattr(fluxes.fastaccess, '_%s_results' % flux.name)
            diff = (results[self.numvars.idx_method] -
                    results[self.numvars.idx_method-1])
            self.numvars.error = max(self.numvars.error,
                                     numpy.max(numpy.abs(diff)))