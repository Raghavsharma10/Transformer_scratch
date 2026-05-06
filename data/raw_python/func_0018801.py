def trim(self, lower=None, upper=None):
        """Trim upper values in accordance with :math:`RelWB \\leq RelWZ`.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(3)
        >>> lnk(ACKER)
        >>> relwz.values = 0.5
        >>> relwb(0.2, 0.5, 0.8)
        >>> relwb
        relwb(0.2, 0.5, 0.5)
        """
        if upper is None:
            upper = getattr(self.subpars.relwz, 'value', None)
        lland_parameters.ParameterSoil.trim(self, lower, upper)