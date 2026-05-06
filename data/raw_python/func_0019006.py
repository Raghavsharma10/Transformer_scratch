def get_sum_fluxes(self):
        """Get the sum of the fluxes calculated so far.

        >>> from hydpy.models.test_v1 import *
        >>> parameterstep()
        >>> fluxes.q = 0.0
        >>> fluxes.fastaccess._q_sum = 1.0
        >>> model.get_sum_fluxes()
        >>> fluxes.q
        q(1.0)
        """
        fluxes = self.sequences.fluxes
        for flux in fluxes.numerics:
            flux(getattr(fluxes.fastaccess, '_%s_sum' % flux.name))