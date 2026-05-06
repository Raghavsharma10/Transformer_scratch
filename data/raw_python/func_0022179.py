def state_province_region(self, value=None):
        """Corresponds to IDD Field `state_province_region`

        Args:
            value (str): value for IDD Field `state_province_region`
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = str(value)
            except ValueError:
                raise ValueError(
                    'value {} need to be of type str '
                    'for field `state_province_region`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `state_province_region`')

        self._state_province_region = value