def reset_sum_fluxes(self):
        """Set the sum of the fluxes calculated so far to zero.

        >>> from hydpy.models.test_v1 import *
        >>> parameterstep()
        >>> fluxes.fastaccess._q_sum = 5.
        >>> model.reset_sum_fluxes()
        >>> fluxes.fastaccess._q_sum
        0.0
        """
        fluxes = self.sequences.fluxes
        for flux in fluxes.numerics:
            if flux.NDIM == 0:
                setattr(fluxes.fastaccess, '_%s_sum' % flux.name, 0.)
            else:
                getattr(fluxes.fastaccess, '_%s_sum' % flux.name)[:] = 0.