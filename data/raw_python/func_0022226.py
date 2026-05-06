def hr_dp010(self, value=None):
        """  Corresponds to IDD Field `hr_dp010`
        humidity ratio corresponding to
        Dew-point temperature corresponding to 1.0,% annual cumulative frequency of occurrence
        calculated at the standard atmospheric pressure at elevation of station

        Args:
            value (float): value for IDD Field `hr_dp010`
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
                                 'for field `hr_dp010`'.format(value))

        self._hr_dp010 = value