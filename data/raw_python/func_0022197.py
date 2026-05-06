def dp990(self, value=None):
        """  Corresponds to IDD Field `dp990`
        Dew-point temperature corresponding to 90.0% annual cumulative
        frequency of occurrence (cold conditions)

        Args:
            value (float): value for IDD Field `dp990`
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
                                 'for field `dp990`'.format(value))

        self._dp990 = value