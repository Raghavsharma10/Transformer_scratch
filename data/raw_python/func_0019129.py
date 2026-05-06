def trim(self, lower=None, upper=None):
        """Trim values in accordance with :math:`WAeS \\leq PWMax \\cdot WATS`,
        or at least in accordance with if :math:`WATS \\geq 0`.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(7)
        >>> pwmax(2.0)
        >>> states.waes = -1., 0., 1., -1., 5., 10., 20.
        >>> states.wats(-1., 0., 0., 5., 5., 5., 5.)
        >>> states.wats
        wats(0.0, 0.0, 0.5, 5.0, 5.0, 5.0, 10.0)
        """
        pwmax = self.subseqs.seqs.model.parameters.control.pwmax
        waes = self.subseqs.waes
        if lower is None:
            lower = numpy.clip(waes/pwmax, 0., numpy.inf)
            lower[numpy.isnan(lower)] = 0.0
        lland_sequences.State1DSequence.trim(self, lower, upper)