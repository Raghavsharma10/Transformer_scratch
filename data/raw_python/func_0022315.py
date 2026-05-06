def relative_humidity(self, value=999):
        """Corresponds to IDD Field `relative_humidity`

        Args:
            value (int): value for IDD Field `relative_humidity`
                value >= 0
                value <= 110
                Missing value: 999
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = int(value)
            except ValueError:
                raise ValueError('value {} need to be of type int '
                                 'for field `relative_humidity`'.format(value))
            if value < 0:
                raise ValueError('value need to be greater or equal 0 '
                                 'for field `relative_humidity`')
            if value > 110:
                raise ValueError('value need to be smaller 110 '
                                 'for field `relative_humidity`')

        self._relative_humidity = value