def depth_soil_specific_heat(self, value=None):
        """Corresponds to IDD Field `depth_soil_specific_heat`

        Args:
            value (float): value for IDD Field `depth_soil_specific_heat`
                Unit: J/kg-K,
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
                    'for field `depth_soil_specific_heat`'.format(value))

        self._depth_soil_specific_heat = value