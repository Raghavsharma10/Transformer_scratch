def design_stat_extremes(self, value="Extremes"):
        """Corresponds to IDD Field `design_stat_extremes`

        Args:
            value (str): value for IDD Field `design_stat_extremes`
                Accepted values are:
                      - Extremes
                Default value: Extremes
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
                    'for field `design_stat_extremes`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `design_stat_extremes`')
            vals = set()
            vals.add("Extremes")
            if value not in vals:
                raise ValueError('value {} is not an accepted value for '
                                 'field `design_stat_extremes`'.format(value))

        self._design_stat_extremes = value