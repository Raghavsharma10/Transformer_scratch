def days_since_last_snowfall(self, value=99):
        """Corresponds to IDD Field `days_since_last_snowfall`

        Args:
            value (int): value for IDD Field `days_since_last_snowfall`
                Missing value: 99
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = int(value)
            except ValueError:
                raise ValueError(
                    'value {} need to be of type int '
                    'for field `days_since_last_snowfall`'.format(value))

        self._days_since_last_snowfall = value