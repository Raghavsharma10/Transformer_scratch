def trim(self, lower=None, upper=None):
        """Trim values in accordance with :math:`WC \\leq WHC \\cdot SP`.

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(7)
        >>> whc(0.1)
        >>> states.wc.values = -1.0, 0.0, 1.0, -1.0, 0.0, 0.5, 1.0
        >>> states.sp(-1., 0., 0., 5., 5., 5., 5.)
        >>> states.sp
        sp(0.0, 0.0, 10.0, 5.0, 5.0, 5.0, 10.0)
        """
        whc = self.subseqs.seqs.model.parameters.control.whc
        wc = self.subseqs.wc
        if lower is None:
            if wc.values is not None:
                with numpy.errstate(divide='ignore', invalid='ignore'):
                    lower = numpy.clip(wc.values / whc.values, 0., numpy.inf)
            else:
                lower = 0.
        hland_sequences.State1DSequence.trim(self, lower, upper)