def db_wb020(self, value=None):
        """  Corresponds to IDD Field `db_wb020`
        mean coincident dry-bulb temperature to
        Wet-bulb temperature corresponding to 2.0% annual cumulative frequency of occurrence

        Args:
            value (float): value for IDD Field `db_wb020`
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
                                 'for field `db_wb020`'.format(value))

        self._db_wb020 = value