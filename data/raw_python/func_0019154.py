def verify(self) -> None:
        """Raise a |RuntimeError| and removes all handled neural networks,
        if the they are defined inconsistently.

        Dispite all automated safety checks explained in the general
        documentation on class |anntools.SeasonalANN|, it is still possible
        to destroy the inner consistency of a |anntools.SeasonalANN| instance,
        as it stores its |anntools.ANN| objects by reference.  This is shown
        by the following example:

        >>> from hydpy import SeasonalANN, ann
        >>> seasonalann = SeasonalANN(None)
        >>> seasonalann.simulationstep = '1d'
        >>> jan = ann(nmb_inputs=1, nmb_neurons=(1,), nmb_outputs=1,
        ...           weights_input=0.0, weights_output=0.0,
        ...           intercepts_hidden=0.0, intercepts_output=1.0)
        >>> seasonalann(_1_1_12=jan)
        >>> jan.nmb_inputs, jan.nmb_outputs = 2, 3
        >>> jan.nmb_inputs, jan.nmb_outputs
        (2, 3)
        >>> seasonalann.nmb_inputs, seasonalann.nmb_outputs
        (1, 1)

        Due to the C level implementation of the mathematical core of both
        |anntools.ANN| and |anntools.SeasonalANN| in module |annutils|,
        such an inconsistency might result in a program crash without any
        informative error message.  Whenever you are afraid some
        inconsistency might have crept in, and you want to find out if this
        is actually the case, call method |anntools.SeasonalANN.verify|
        explicitly:

        >>> seasonalann.verify()
        Traceback (most recent call last):
        ...
        RuntimeError: The number of input and output values of all neural \
networks contained by a seasonal neural network collection must be \
identical and be known by the containing object.  But the seasonal \
neural network collection `seasonalann` of element `?` assumes `1` input \
and `1` output values, while the network corresponding to the time of \
year `toy_1_1_12_0_0` requires `2` input and `3` output values.

        >>> seasonalann
        seasonalann()

        >>> seasonalann.verify()
        Traceback (most recent call last):
        ...
        RuntimeError: Seasonal artificial neural network collections need \
to handle at least one "normal" single neural network, but for the seasonal \
neural network `seasonalann` of element `?` none has been defined so far.
        """
        if not self.anns:
            self._toy2ann.clear()
            raise RuntimeError(
                'Seasonal artificial neural network collections need '
                'to handle at least one "normal" single neural network, '
                'but for the seasonal neural network `%s` of element '
                '`%s` none has been defined so far.'
                % (self.name, objecttools.devicename(self)))
        for toy, ann_ in self:
            ann_.verify()
            if ((self.nmb_inputs != ann_.nmb_inputs) or
                    (self.nmb_outputs != ann_.nmb_outputs)):
                self._toy2ann.clear()
                raise RuntimeError(
                    'The number of input and output values of all neural '
                    'networks contained by a seasonal neural network '
                    'collection must be identical and be known by the '
                    'containing object.  But the seasonal neural '
                    'network collection `%s` of element `%s` assumes '
                    '`%d` input and `%d` output values, while the network '
                    'corresponding to the time of year `%s` requires '
                    '`%d` input and `%d` output values.'
                    % (self.name, objecttools.devicename(self),
                       self.nmb_inputs, self.nmb_outputs,
                       toy,
                       ann_.nmb_inputs, ann_.nmb_outputs))