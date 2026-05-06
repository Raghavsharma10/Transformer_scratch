def data_period_start_day_of_week(self, value=None):
        """Corresponds to IDD Field `data_period_start_day_of_week`

        Args:
            value (str): value for IDD Field `data_period_start_day_of_week`
                Accepted values are:
                      - Sunday
                      - Monday
                      - Tuesday
                      - Wednesday
                      - Thursday
                      - Friday
                      - Saturday
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
                    'for field `data_period_start_day_of_week`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `data_period_start_day_of_week`')
            vals = set()
            vals.add("Sunday")
            vals.add("Monday")
            vals.add("Tuesday")
            vals.add("Wednesday")
            vals.add("Thursday")
            vals.add("Friday")
            vals.add("Saturday")
            if value not in vals:
                raise ValueError(
                    'value {} is not an accepted value for '
                    'field `data_period_start_day_of_week`'.format(value))

        self._data_period_start_day_of_week = value