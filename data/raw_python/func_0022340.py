def liquid_precipitation_depth(self, value=999.0):
        """Corresponds to IDD Field `liquid_precipitation_depth`

        Args:
            value (float): value for IDD Field `liquid_precipitation_depth`
                Unit: mm
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
                raise ValueError(
                    'value {} need to be of type float '
                    'for field `liquid_precipitation_depth`'.format(value))

        self._liquid_precipitation_depth = value