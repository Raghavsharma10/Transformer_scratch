def dbmax50years(self, value=None):
        """  Corresponds to IDD Field `dbmax50years`
        50-year return period values for maximum extreme dry-bulb temperature

        Args:
            value (float): value for IDD Field `dbmax50years`
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
                                 'for field `dbmax50years`'.format(value))

        self._dbmax50years = value