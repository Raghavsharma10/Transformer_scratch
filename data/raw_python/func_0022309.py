def day(self, value=None):
        """Corresponds to IDD Field `day`

        Args:
            value (int): value for IDD Field `day`
                value >= 1
                value <= 31
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
                                 'for field `day`'.format(value))
            if value < 1:
                raise ValueError('value need to be greater or equal 1 '
                                 'for field `day`')
            if value > 31:
                raise ValueError('value need to be smaller 31 '
                                 'for field `day`')

        self._day = value