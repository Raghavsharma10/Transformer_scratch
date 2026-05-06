def zenith_luminance(self, value=9999.0):
        """  Corresponds to IDD Field `zenith_luminance`
        will be missing if >= 9999

        Args:
            value (float): value for IDD Field `zenith_luminance`
                Unit: Cd/m2
                value >= 0.0
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
                                 'for field `zenith_luminance`'.format(value))
            if value < 0.0:
                raise ValueError('value need to be greater or equal 0.0 '
                                 'for field `zenith_luminance`')

        self._zenith_luminance = value