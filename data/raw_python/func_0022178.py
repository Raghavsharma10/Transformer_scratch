def city(self, value=None):
        """Corresponds to IDD Field `city`

        Args:
            value (str): value for IDD Field `city`
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = str(value)
            except ValueError:
                raise ValueError('value {} need to be of type str '
                                 'for field `city`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `city`')

        self._city = value