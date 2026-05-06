def wb020(self, value=None):
        """  Corresponds to IDD Field `wb020`
        Wet-bulb temperature corresponding to 02.0% annual cumulative frequency of occurrence

        Args:
            value (float): value for IDD Field `wb020`
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
                                 'for field `wb020`'.format(value))

        self._wb020 = value