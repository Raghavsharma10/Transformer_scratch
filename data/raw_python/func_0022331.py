def visibility(self, value=9999.0):
        """Corresponds to IDD Field `visibility` This is the value for
        visibility in km. (Horizontal visibility at the time indicated.)

        Args:
            value (float): value for IDD Field `visibility`
                Unit: km
                Missing value: 9999.0
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
                                 'for field `visibility`'.format(value))

        self._visibility = value