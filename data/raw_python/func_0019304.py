def trim(self, lower=None, upper=None):
        """Trim negative value whenever there is no internal lake within
        the respective subbasin.

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(2)
        >>> zonetype(FIELD, ILAKE)
        >>> states.lz(-1.0)
        >>> states.lz
        lz(-1.0)
        >>> zonetype(FIELD, FOREST)
        >>> states.lz(-1.0)
        >>> states.lz
        lz(0.0)
        >>> states.lz(1.0)
        >>> states.lz
        lz(1.0)
        """
        if upper is None:
            control = self.subseqs.seqs.model.parameters.control
            if not any(control.zonetype.values == ILAKE):
                lower = 0.
        sequencetools.StateSequence.trim(self, lower, upper)