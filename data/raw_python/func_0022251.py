def dbmin20years(self, value=None):
        """  Corresponds to IDD Field `dbmin20years`
        20-year return period values for minimum extreme dry-bulb temperature

        Args:
            value (float): value for IDD Field `dbmin20years`
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
                                 'for field `dbmin20years`'.format(value))

        self._dbmin20years = value