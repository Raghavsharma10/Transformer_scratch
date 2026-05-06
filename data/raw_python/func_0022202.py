def ws010c(self, value=None):
        """  Corresponds to IDD Field `ws010c`
        Wind speed corresponding to 1.0% cumulative frequency
        of occurrence for coldest month;

        Args:
            value (float): value for IDD Field `ws010c`
                Unit: m/s
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
                                 'for field `ws010c`'.format(value))

        self._ws010c = value