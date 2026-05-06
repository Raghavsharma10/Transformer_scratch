def precipitable_water(self, value=999.0):
        """Corresponds to IDD Field `precipitable_water`

        Args:
            value (float): value for IDD Field `precipitable_water`
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
                    'for field `precipitable_water`'.format(value))

        self._precipitable_water = value