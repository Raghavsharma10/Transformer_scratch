def month(self, value=None):
        """Corresponds to IDD Field `month`

        Args:
            value (int): value for IDD Field `month`
                value >= 1
                value <= 12
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
                                 'for field `month`'.format(value))
            if value < 1:
                raise ValueError('value need to be greater or equal 1 '
                                 'for field `month`')
            if value > 12:
                raise ValueError('value need to be smaller 12 '
                                 'for field `month`')

        self._month = value