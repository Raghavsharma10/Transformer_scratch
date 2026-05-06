def atmospheric_station_pressure(self, value=999999):
        """Corresponds to IDD Field `atmospheric_station_pressure`

        Args:
            value (int): value for IDD Field `atmospheric_station_pressure`
                Unit: Pa
                value > 31000
                value < 120000
                Missing value: 999999
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = int(value)
            except ValueError:
                raise ValueError(
                    'value {} need to be of type int '
                    'for field `atmospheric_station_pressure`'.format(value))
            if value <= 31000:
                raise ValueError('value need to be greater 31000 '
                                 'for field `atmospheric_station_pressure`')
            if value >= 120000:
                raise ValueError('value need to be smaller 120000 '
                                 'for field `atmospheric_station_pressure`')

        self._atmospheric_station_pressure = value