def ws_db004(self, value=None):
        """  Corresponds to IDD Field `ws_db004`
        Mean wind speed coincident with 0.4% dry-bulb temperature

        Args:
            value (float): value for IDD Field `ws_db004`
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
                                 'for field `ws_db004`'.format(value))

        self._ws_db004 = value