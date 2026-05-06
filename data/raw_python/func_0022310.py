def hour(self, value=None):
        """Corresponds to IDD Field `hour`

        Args:
            value (int): value for IDD Field `hour`
                value >= 1
                value <= 24
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
                                 'for field `hour`'.format(value))
            if value < 1:
                raise ValueError('value need to be greater or equal 1 '
                                 'for field `hour`')
            if value > 24:
                raise ValueError('value need to be smaller 24 '
                                 'for field `hour`')

        self._hour = value