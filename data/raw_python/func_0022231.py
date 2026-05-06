def en004(self, value=None):
        """  Corresponds to IDD Field `en004`
        mean coincident dry-bulb temperature to
        Enthalpy corresponding to 0.4% annual cumulative frequency of occurrence

        Args:
            value (float): value for IDD Field `en004`
                Unit: kJ/kg
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
                                 'for field `en004`'.format(value))

        self._en004 = value