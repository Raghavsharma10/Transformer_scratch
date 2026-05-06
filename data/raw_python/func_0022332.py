def ceiling_height(self, value=99999.0):
        """Corresponds to IDD Field `ceiling_height` This is the value for
        ceiling height in m. (77777 is unlimited ceiling height. 88888 is
        cirroform ceiling.) It is not currently used in EnergyPlus
        calculations.

        Args:
            value (float): value for IDD Field `ceiling_height`
                Unit: m
                Missing value: 99999.0
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = float(value)
            except ValueError:
                raise ValueError('value {} need to be of type float '
                                 'for field `ceiling_height`'.format(value))

        self._ceiling_height = value