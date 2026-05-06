def hr_dp990(self, value=None):
        """  Corresponds to IDD Field `hr_dp990`
        humidity ratio, calculated at standard atmospheric pressure
        at elevation of station, corresponding to
        Dew-point temperature corresponding to 90.0% annual cumulative
        frequency of occurrence (cold conditions)

        Args:
            value (float): value for IDD Field `hr_dp990`
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
                                 'for field `hr_dp990`'.format(value))

        self._hr_dp990 = value