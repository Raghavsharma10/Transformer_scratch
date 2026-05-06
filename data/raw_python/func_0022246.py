def dbmax_stddev(self, value=None):
        """  Corresponds to IDD Field `dbmax_stddev`
        Standard deviation of extreme annual maximum dry-bulb temperature

        Args:
            value (float): value for IDD Field `dbmax_stddev`
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
                                 'for field `dbmax_stddev`'.format(value))

        self._dbmax_stddev = value