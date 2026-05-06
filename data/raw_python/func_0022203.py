def db_ws010c(self, value=None):
        """  Corresponds to IDD Field `db_ws010c`
        Mean coincident dry-bulb temperature to wind speed corresponding to 1.0% cumulative frequency for coldest month

        Args:
            value (float): value for IDD Field `db_ws010c`
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
                                 'for field `db_ws010c`'.format(value))

        self._db_ws010c = value