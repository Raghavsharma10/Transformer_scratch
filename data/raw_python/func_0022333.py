def present_weather_observation(self, value=None):
        """Corresponds to IDD Field `present_weather_observation` If the value
        of the field is 0, then the observed weather codes are taken from the
        following field. If the value of the field is 9, then "missing" weather
        is assumed. Since the primary use of these fields (Present Weather
        Observation and Present Weather Codes) is for rain/wet surfaces, a
        missing observation field or a missing weather code implies "no rain".

        Args:
            value (int): value for IDD Field `present_weather_observation`
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
                    'for field `present_weather_observation`'.format(value))

        self._present_weather_observation = value