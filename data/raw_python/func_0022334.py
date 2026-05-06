def present_weather_codes(self, value=None):
        """Corresponds to IDD Field `present_weather_codes`

        Args:
            value (int): value for IDD Field `present_weather_codes`
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
                    'for field `present_weather_codes`'.format(value))

        self._present_weather_codes = value