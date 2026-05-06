def db990(self, value=None):
        """  Corresponds to IDD Field `db990`
        Dry-bulb temperature corresponding to 90.0% annual cumulative
        frequency of occurrence (cold conditions)

        Args:
            value (float): value for IDD Field `db990`
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
                                 'for field `db990`'.format(value))

        self._db990 = value