def coldestmonth(self, value=None):
        """Corresponds to IDD Field `coldestmonth`

        Args:
            value (int): value for IDD Field `coldestmonth`
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
                                 'for field `coldestmonth`'.format(value))
            if value < 1:
                raise ValueError('value need to be greater or equal 1 '
                                 'for field `coldestmonth`')
            if value > 12:
                raise ValueError('value need to be smaller 12 '
                                 'for field `coldestmonth`')

        self._coldestmonth = value