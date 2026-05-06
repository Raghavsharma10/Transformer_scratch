def dbr(self, value=None):
        """Corresponds to IDD Field `dbr` Daily temperature range for hottest
        month.

        [defined as mean of the difference between daily maximum
        and daily minimum dry-bulb temperatures for hottest month]

        Args:
            value (float): value for IDD Field `dbr`
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
                                 'for field `dbr`'.format(value))

        self._dbr = value