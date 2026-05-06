def dbmax10years(self, value=None):
        """  Corresponds to IDD Field `dbmax10years`
        10-year return period values for maximum extreme dry-bulb temperature

        Args:
            value (float): value for IDD Field `dbmax10years`
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
                                 'for field `dbmax10years`'.format(value))

        self._dbmax10years = value