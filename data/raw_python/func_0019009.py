def addup_fluxes(self):
        """Add up the sum of the fluxes calculated so far.

        >>> from hydpy.models.test_v1 import *
        >>> parameterstep()
        >>> fluxes.fastaccess._q_sum = 1.0
        >>> fluxes.q(2.0)
        >>> model.addup_fluxes()
        >>> fluxes.fastaccess._q_sum
        3.0
        """
        fluxes = self.sequences.fluxes
        for flux in fluxes.numerics:
            sum_ = getattr(fluxes.fastaccess, '_%s_sum' % flux.name)
            sum_ += flux
            if flux.NDIM == 0:
                setattr(fluxes.fastaccess, '_%s_sum' % flux.name, sum_)