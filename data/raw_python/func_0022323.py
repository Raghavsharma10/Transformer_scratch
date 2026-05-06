def global_horizontal_illuminance(self, value=999999.0):
        """  Corresponds to IDD Field `global_horizontal_illuminance`
        will be missing if >= 999900

        Args:
            value (float): value for IDD Field `global_horizontal_illuminance`
                Unit: lux
                value >= 0.0
                Missing value: 999999.0
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
                    'for field `global_horizontal_illuminance`'.format(value))
            if value < 0.0:
                raise ValueError('value need to be greater or equal 0.0 '
                                 'for field `global_horizontal_illuminance`')

        self._global_horizontal_illuminance = value