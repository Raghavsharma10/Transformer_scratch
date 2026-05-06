def refresh(self) -> None:
        """Prepare the actual |anntools.SeasonalANN| object for calculations.

        Dispite all automated refreshings explained in the general
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

        Due to the C level implementation of the mathematical core of
        both |anntools.ANN| and |anntools.SeasonalANN| in module |annutils|,
        such an inconsistency might result in a program crash without any
        informative error message.  Whenever you are afraid some
        inconsistency might have crept in, and you want to repair it,
        call method |anntools.SeasonalANN.refresh| explicitly:

        >>> seasonalann.refresh()
        >>> jan.nmb_inputs, jan.nmb_outputs
        (2, 3)
        >>> seasonalann.nmb_inputs, seasonalann.nmb_outputs
        (2, 3)
        """
        # pylint: disable=unsupported-assignment-operation
        if self._do_refresh:
            if self.anns:
                self.__sann = annutils.SeasonalANN(self.anns)
                setattr(self.fastaccess, self.name, self._sann)
                self._set_shape((None, self._sann.nmb_anns))
                if self._sann.nmb_anns > 1:
                    self._interp()
                else:
                    self._sann.ratios[:, 0] = 1.
                self.verify()
            else:
                self.__sann = None