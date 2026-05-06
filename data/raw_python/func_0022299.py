def number_of_records_per_hour(self, value=None):
        """Corresponds to IDD Field `number_of_records_per_hour`

        Args:
            value (int): value for IDD Field `number_of_records_per_hour`
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
                    'for field `number_of_records_per_hour`'.format(value))

        self._number_of_records_per_hour = value