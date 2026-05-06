def ws004c(self, value=None):
        """Corresponds to IDD Field `ws004c`

        Args:
            value (float): value for IDD Field `ws004c`
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
                                 'for field `ws004c`'.format(value))

        self._ws004c = value