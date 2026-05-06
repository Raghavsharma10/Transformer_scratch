def dew_point_temperature(self, value=99.9):
        """Corresponds to IDD Field `dew_point_temperature`

        Args:
            value (float): value for IDD Field `dew_point_temperature`
                Unit: C
                value > -70.0
                value < 70.0
                Missing value: 99.9
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = float(value)
            except ValueError:
                raise ValueError(
                    'value {} need to be of type float '
                    'for field `dew_point_temperature`'.format(value))
            if value <= -70.0:
                raise ValueError('value need to be greater -70.0 '
                                 'for field `dew_point_temperature`')
            if value >= 70.0:
                raise ValueError('value need to be smaller 70.0 '
                                 'for field `dew_point_temperature`')

        self._dew_point_temperature = value