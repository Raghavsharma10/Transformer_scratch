def title_of_design_condition(self, value=None):
        """Corresponds to IDD Field `title_of_design_condition`

        Args:
            value (str): value for IDD Field `title_of_design_condition`
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
                    'for field `title_of_design_condition`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `title_of_design_condition`')

        self._title_of_design_condition = value