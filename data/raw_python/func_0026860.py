def _validate_gain_A_value(self, gain_A):
        """
        validate a given value for gain_A

        :type gain_A: int
        :raises: ValueError
        """
        if gain_A not in self._valid_gains_for_channel_A:
            raise ParameterValidationError("{gain_A} is not a valid gain".format(gain_A=gain_A))