def trim(self, lower=None, upper=None):
        """Trim values in accordance with :math:`WAeS \\leq PWMax \\cdot WATS`.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(7)
        >>> pwmax(2.)
        >>> states.wats = 0., 0., 0., 5., 5., 5., 5.
        >>> states.waes(-1., 0., 1., -1., 5., 10., 20.)
        >>> states.waes
        waes(0.0, 0.0, 0.0, 0.0, 5.0, 10.0, 10.0)
        """
        pwmax = self.subseqs.seqs.model.parameters.control.pwmax
        wats = self.subseqs.wats
        if upper is None:
            upper = pwmax*wats
        lland_sequences.State1DSequence.trim(self, lower, upper)