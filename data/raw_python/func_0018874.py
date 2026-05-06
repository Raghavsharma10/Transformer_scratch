def refweights(self):
        """A |numpy| |numpy.ndarray| with equal weights for all segment
        junctions..

        >>> from hydpy.models.hstream import *
        >>> parameterstep('1d')
        >>> states.qjoints.shape = 5
        >>> states.qjoints.refweights
        array([ 0.2,  0.2,  0.2,  0.2,  0.2])
        """
        # pylint: disable=unsubscriptable-object
        # due to a pylint bug (see https://github.com/PyCQA/pylint/issues/870)
        return numpy.full(self.shape, 1./self.shape[0], dtype=float)