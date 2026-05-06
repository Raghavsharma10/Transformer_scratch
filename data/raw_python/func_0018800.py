def trim(self, lower=None, upper=None):
        """Trim upper values in accordance with :math:`RelWB \\leq RelWZ`.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(3)
        >>> lnk(ACKER)
        >>> relwb.values = 0.5
        >>> relwz(0.2, 0.5, 0.8)
        >>> relwz
        relwz(0.5, 0.5, 0.8)
        """
        if lower is None:
            lower = getattr(self.subpars.relwb, 'value', None)
        lland_parameters.ParameterSoil.trim(self, lower, upper)