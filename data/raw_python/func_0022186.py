def elevation(self, value=0.0):
        """Corresponds to IDD Field `elevation`

        Args:
            value (float): value for IDD Field `elevation`
                Unit: m
                Default value: 0.0
                value >= -1000.0
                value < 9999.9
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
                                 'for field `elevation`'.format(value))
            if value < -1000.0:
                raise ValueError('value need to be greater or equal -1000.0 '
                                 'for field `elevation`')
            if value >= 9999.9:
                raise ValueError('value need to be smaller 9999.9 '
                                 'for field `elevation`')

        self._elevation = value