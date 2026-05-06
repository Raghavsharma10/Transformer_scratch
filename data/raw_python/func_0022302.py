def data_period_start_day(self, value=None):
        """Corresponds to IDD Field `data_period_start_day`

        Args:
            value (str): value for IDD Field `data_period_start_day`
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = str(value)
            except ValueError:
                raise ValueError(
                    'value {} need to be of type str '
                    'for field `data_period_start_day`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `data_period_start_day`')

        self._data_period_start_day = value