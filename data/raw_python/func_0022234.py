def db_en010(self, value=None):
        """  Corresponds to IDD Field `db_en010`
        mean coincident dry-bulb temperature to
        Enthalpy corresponding to 1.0% annual cumulative frequency of occurrence

        Args:
            value (float): value for IDD Field `db_en010`
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
                                 'for field `db_en010`'.format(value))

        self._db_en010 = value