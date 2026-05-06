def design_stat_heating(self, value="Heating"):
        """Corresponds to IDD Field `design_stat_heating`

        Args:
            value (str): value for IDD Field `design_stat_heating`
                Accepted values are:
                      - Heating
                Default value: Heating
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
                    'for field `design_stat_heating`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `design_stat_heating`')
            vals = set()
            vals.add("Heating")
            if value not in vals:
                raise ValueError('value {} is not an accepted value for '
                                 'field `design_stat_heating`'.format(value))

        self._design_stat_heating = value