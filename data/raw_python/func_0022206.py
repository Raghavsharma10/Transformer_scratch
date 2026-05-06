def design_stat_cooling(self, value="Cooling"):
        """Corresponds to IDD Field `design_stat_cooling`

        Args:
            value (str): value for IDD Field `design_stat_cooling`
                Accepted values are:
                      - Cooling
                Default value: Cooling
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
                    'for field `design_stat_cooling`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `design_stat_cooling`')
            vals = set()
            vals.add("Cooling")
            if value not in vals:
                raise ValueError('value {} is not an accepted value for '
                                 'field `design_stat_cooling`'.format(value))

        self._design_stat_cooling = value