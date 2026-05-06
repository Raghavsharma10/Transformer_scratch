def wmo(self, value=None):
        """Corresponds to IDD Field `wmo` usually a 6 digit field. Used as
        alpha in EnergyPlus.

        Args:
            value (str): value for IDD Field `wmo`
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
                                 'for field `wmo`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `wmo`')

        self._wmo = value