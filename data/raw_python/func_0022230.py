def db_dp020(self, value=None):
        """  Corresponds to IDD Field `db_dp020`
        mean coincident dry-bulb temperature to
        Dew-point temperature corresponding to 2.0% annual cumulative frequency of occurrence

        Args:
            value (float): value for IDD Field `db_dp020`
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
                                 'for field `db_dp020`'.format(value))

        self._db_dp020 = value