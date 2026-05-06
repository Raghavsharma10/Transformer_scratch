def trim(self, lower=None, upper=None):
        """Trim upper values in accordance with :math:`IC \\leq ICMAX`.

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(5)
        >>> icmax(2.0)
        >>> states.ic(-1.0, 0.0, 1.0, 2.0, 3.0)
        >>> states.ic
        ic(0.0, 0.0, 1.0, 2.0, 2.0)
        """
        if upper is None:
            control = self.subseqs.seqs.model.parameters.control
            upper = control.icmax
        hland_sequences.State1DSequence.trim(self, lower, upper)