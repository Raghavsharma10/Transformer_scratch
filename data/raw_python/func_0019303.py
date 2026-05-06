def trim(self, lower=None, upper=None):
        """Trim values in accordance with :math:`WC \\leq WHC \\cdot SP`.

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(7)
        >>> whc(0.1)
        >>> states.sp = 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0
        >>> states.wc(-1.0, 0.0, 1.0, -1.0, 0.0, 0.5, 1.0)
        >>> states.wc
        wc(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5)
        """
        whc = self.subseqs.seqs.model.parameters.control.whc
        sp = self.subseqs.sp
        if (upper is None) and (sp.values is not None):
            upper = whc*sp
        hland_sequences.State1DSequence.trim(self, lower, upper)