def trim(self, lower=None, upper=None):
        """Trim upper values in accordance with
        :math:`EQI2 \\leq EQI1 \\leq EQB`.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> eqb.value = 3.0
        >>> eqi2.value = 1.0
        >>> eqi1(0.0)
        >>> eqi1
        eqi1(1.0)
        >>> eqi1(1.0)
        >>> eqi1
        eqi1(1.0)
        >>> eqi1(2.0)
        >>> eqi1
        eqi1(2.0)
        >>> eqi1(3.0)
        >>> eqi1
        eqi1(3.0)
        >>> eqi1(4.0)
        >>> eqi1
        eqi1(3.0)
        """
        if lower is None:
            lower = getattr(self.subpars.eqi2, 'value', None)
        if upper is None:
            upper = getattr(self.subpars.eqb, 'value', None)
        super().trim(lower, upper)