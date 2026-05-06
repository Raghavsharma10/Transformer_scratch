def trim(self, lower=None, upper=None):
        """Trim values in accordance with :math:`BoWa \\leq NFk`.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(5)
        >>> nfk(200.)
        >>> states.bowa(-100.,0., 100., 200., 300.)
        >>> states.bowa
        bowa(0.0, 0.0, 100.0, 200.0, 200.0)
        """
        if upper is None:
            upper = self.subseqs.seqs.model.parameters.control.nfk
        lland_sequences.State1DSequence.trim(self, lower, upper)