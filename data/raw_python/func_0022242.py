def wbmax(self, value=None):
        """  Corresponds to IDD Field `wbmax`
        Extreme maximum wet-bulb temperature

        Args:
            value (float): value for IDD Field `wbmax`
                Unit: C
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
                                 'for field `wbmax`'.format(value))

        self._wbmax = value