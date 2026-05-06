def trim(self, lower=None, upper=None):
        """Trim upper values in accordance with :math:`EQI1 \\leq EQB`.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> eqi1.value = 2.0
        >>> eqb(1.0)
        >>> eqb
        eqb(2.0)
        >>> eqb(2.0)
        >>> eqb
        eqb(2.0)
        >>> eqb(3.0)
        >>> eqb
        eqb(3.0)
        """
        if lower is None:
            lower = getattr(self.subpars.eqi1, 'value', None)
        super().trim(lower, upper)