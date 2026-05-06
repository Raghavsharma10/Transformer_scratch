def minute(self, value=None):
        """Corresponds to IDD Field `minute`

        Args:
            value (int): value for IDD Field `minute`
                value >= 0
                value <= 60
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
                                 'for field `minute`'.format(value))
            if value < 0:
                raise ValueError('value need to be greater or equal 0 '
                                 'for field `minute`')
            if value > 60:
                raise ValueError('value need to be smaller 60 '
                                 'for field `minute`')

        self._minute = value