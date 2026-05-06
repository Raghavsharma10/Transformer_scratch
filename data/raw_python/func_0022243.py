def dbmin_mean(self, value=None):
        """  Corresponds to IDD Field `dbmin_mean`
        Mean of extreme annual minimum dry-bulb temperature

        Args:
            value (float): value for IDD Field `dbmin_mean`
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
                                 'for field `dbmin_mean`'.format(value))

        self._dbmin_mean = value