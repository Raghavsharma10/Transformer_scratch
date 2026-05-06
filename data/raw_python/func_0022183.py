def latitude(self, value=0.0):
        """Corresponds to IDD Field `latitude`

        + is North, - is South, degree minutes represented in decimal (i.e. 30 minutes is .5)

        Args:
            value (float): value for IDD Field `latitude`
                Unit: deg
                Default value: 0.0
                value >= -90.0
                value <= 90.0
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
                                 'for field `latitude`'.format(value))
            if value < -90.0:
                raise ValueError('value need to be greater or equal -90.0 '
                                 'for field `latitude`')
            if value > 90.0:
                raise ValueError('value need to be smaller 90.0 '
                                 'for field `latitude`')

        self._latitude = value