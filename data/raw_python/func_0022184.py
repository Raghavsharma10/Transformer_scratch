def longitude(self, value=0.0):
        """Corresponds to IDD Field `longitude`

        - is West, + is East, degree minutes represented in decimal (i.e. 30 minutes is .5)

        Args:
            value (float): value for IDD Field `longitude`
                Unit: deg
                Default value: 0.0
                value >= -180.0
                value <= 180.0
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
                                 'for field `longitude`'.format(value))
            if value < -180.0:
                raise ValueError('value need to be greater or equal -180.0 '
                                 'for field `longitude`')
            if value > 180.0:
                raise ValueError('value need to be smaller 180.0 '
                                 'for field `longitude`')

        self._longitude = value