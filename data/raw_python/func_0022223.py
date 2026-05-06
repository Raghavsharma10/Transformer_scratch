def hr_dp004(self, value=None):
        """  Corresponds to IDD Field `hr_dp004`
        humidity ratio corresponding to
        Dew-point temperature corresponding to 0.4% annual cumulative frequency of occurrence

        Args:
            value (float): value for IDD Field `hr_dp004`
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
                                 'for field `hr_dp004`'.format(value))

        self._hr_dp004 = value