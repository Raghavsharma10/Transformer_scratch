def timezone(self, value=0.0):
        """Corresponds to IDD Field `timezone` Time relative to GMT.

        Args:
            value (float): value for IDD Field `timezone`
                Unit: hr - not on standard units list???
                Default value: 0.0
                value >= -12.0
                value <= 12.0
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = float(value)
            except ValueError:
                raise ValueError('value {} need to be of type float '
                                 'for field `timezone`'.format(value))
            if value < -12.0:
                raise ValueError('value need to be greater or equal -12.0 '
                                 'for field `timezone`')
            if value > 12.0:
                raise ValueError('value need to be smaller 12.0 '
                                 'for field `timezone`')

        self._timezone = value