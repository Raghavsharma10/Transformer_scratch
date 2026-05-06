def wind_speed(self, value=999.0):
        """Corresponds to IDD Field `wind_speed`

        Args:
            value (float): value for IDD Field `wind_speed`
                Unit: m/s
                value >= 0.0
                value <= 40.0
                Missing value: 999.0
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
                                 'for field `wind_speed`'.format(value))
            if value < 0.0:
                raise ValueError('value need to be greater or equal 0.0 '
                                 'for field `wind_speed`')
            if value > 40.0:
                raise ValueError('value need to be smaller 40.0 '
                                 'for field `wind_speed`')

        self._wind_speed = value