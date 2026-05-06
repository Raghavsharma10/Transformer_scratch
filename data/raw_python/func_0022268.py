def depth_soil_density(self, value=None):
        """Corresponds to IDD Field `depth_soil_density`

        Args:
            value (float): value for IDD Field `depth_soil_density`
                Unit: kg/m3
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
                    'for field `depth_soil_density`'.format(value))

        self._depth_soil_density = value