def ground_temperature_depth(self, value=None):
        """Corresponds to IDD Field `ground_temperature_depth`

        Args:
            value (float): value for IDD Field `ground_temperature_depth`
                Unit: m
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
                    'for field `ground_temperature_depth`'.format(value))

        self._ground_temperature_depth = value