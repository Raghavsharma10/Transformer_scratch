def total_sky_cover(self, value=99.0):
        """Corresponds to IDD Field `total_sky_cover` This is the value for
        total sky cover (tenths of coverage). (i.e. 1 is 1/10 covered. 10 is
        total coverage). (Amount of sky dome in tenths covered by clouds or
        obscuring phenomena at the  hour indicated at the time indicated.)

        Args:
            value (float): value for IDD Field `total_sky_cover`
                value >= 0.0
                value <= 10.0
                Missing value: 99.0
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
                                 'for field `total_sky_cover`'.format(value))
            if value < 0.0:
                raise ValueError('value need to be greater or equal 0.0 '
                                 'for field `total_sky_cover`')
            if value > 10.0:
                raise ValueError('value need to be smaller 10.0 '
                                 'for field `total_sky_cover`')

        self._total_sky_cover = value