def integrate_fluxes(self):
        """Perform a dot multiplication between the fluxes and the
        A coefficients associated with the different stages of the
        actual method.

        >>> from hydpy.models.test_v1 import *
        >>> parameterstep()
        >>> model.numvars.idx_method = 2
        >>> model.numvars.idx_stage = 1
        >>> model.numvars.dt = 0.5
        >>> points = numpy.asarray(fluxes.fastaccess._q_points)
        >>> points[:4] = 15., 2., -999., 0.
        >>> model.integrate_fluxes()
        >>> from hydpy import round_
        >>> from hydpy import pub
        >>> round_(numpy.asarray(model.numconsts.a_coefs)[1, 1, :2])
        0.375, 0.125
        >>> fluxes.q
        q(2.9375)
        """
        fluxes = self.sequences.fluxes
        for flux in fluxes.numerics:
            points = getattr(fluxes.fastaccess, '_%s_points' % flux.name)
            coefs = self.numconsts.a_coefs[self.numvars.idx_method-1,
                                           self.numvars.idx_stage,
                                           :self.numvars.idx_method]
            flux(self.numvars.dt *
                 numpy.dot(coefs, points[:self.numvars.idx_method]))