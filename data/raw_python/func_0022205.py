def wd_db996(self, value=None):
        """  Corresponds to IDD Field `wd_db996`
        most frequent wind direction corresponding to mean wind speed coincident with 99.6% dry-bulb temperature
        degrees from north (east = 90 deg)

        Args:
            value (float): value for IDD Field `wd_db996`
                Unit: deg
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
                                 'for field `wd_db996`'.format(value))

        self._wd_db996 = value