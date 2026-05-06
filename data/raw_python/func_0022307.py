def year(self, value=None):
        """Corresponds to IDD Field `year`

        Args:
            value (int): value for IDD Field `year`
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
                                 'for field `year`'.format(value))

        self._year = value