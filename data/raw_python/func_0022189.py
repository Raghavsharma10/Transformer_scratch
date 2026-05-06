def unkown_field(self, value=None):
        """Corresponds to IDD Field `unkown_field` Empty field in data.

        Args:
            value (str): value for IDD Field `unkown_field`
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
                                 'for field `unkown_field`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `unkown_field`')

        self._unkown_field = value