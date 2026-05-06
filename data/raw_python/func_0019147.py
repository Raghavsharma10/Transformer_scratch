def nmb_neurons(self) -> Tuple[int, ...]:
        """Number of neurons of the hidden layers.

        >>> from hydpy import ANN
        >>> ann = ANN(None)
        >>> ann(nmb_inputs=2, nmb_neurons=(2, 1), nmb_outputs=3)
        >>> ann.nmb_neurons
        (2, 1)
        >>> ann.nmb_neurons = (3,)
        >>> ann.nmb_neurons
        (3,)
        >>> del ann.nmb_neurons
        >>> ann.nmb_neurons
        Traceback (most recent call last):
        ...
        hydpy.core.exceptiontools.AttributeNotReady: Attribute `nmb_neurons` \
of object `ann` has not been prepared so far.
        """
        return tuple(numpy.asarray(self._cann.nmb_neurons))