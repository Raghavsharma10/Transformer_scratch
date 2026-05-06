def depth_december_average_ground_temperature(self, value=None):
        """Corresponds to IDD Field `depth_december_average_ground_temperature`

        Args:
            value (float): value for IDD Field `depth_december_average_ground_temperature`
                Unit: C
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
                    'for field `depth_december_average_ground_temperature`'.format(value))

        self._depth_december_average_ground_temperature = value