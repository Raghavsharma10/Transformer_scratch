def hrs_84_and_db12_8_or_20_6(self, value=None):
        """  Corresponds to IDD Field `hrs_84_and_db12_8_or_20_6`
        Number of hours between 8 AM and 4 PM (inclusive) with dry-bulb temperature between 12.8 and 20.6 C

        Args:
            value (float): value for IDD Field `hrs_84_and_db12_8_or_20_6`
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value
        """
        if value is not None:
            try:
                value = float(value)
            except ValueError:
                raise ValueError(
                    'value {} need to be of type float '
                    'for field `hrs_84_and_db12_8_or_20_6`'.format(value))

        self._hrs_84_and_db12_8_or_20_6 = value