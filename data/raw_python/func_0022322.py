def diffuse_horizontal_radiation(self, value=9999.0):
        """Corresponds to IDD Field `diffuse_horizontal_radiation`

        Args:
            value (float): value for IDD Field `diffuse_horizontal_radiation`
                Unit: Wh/m2
                value >= 0.0
                Missing value: 9999.0
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
                    'for field `diffuse_horizontal_radiation`'.format(value))
            if value < 0.0:
                raise ValueError('value need to be greater or equal 0.0 '
                                 'for field `diffuse_horizontal_radiation`')

        self._diffuse_horizontal_radiation = value