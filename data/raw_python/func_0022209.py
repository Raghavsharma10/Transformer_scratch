def wb_db004(self, value=None):
        """  Corresponds to IDD Field `wb_db004`
        mean coincident wet-bulb temperature to
        Dry-bulb temperature corresponding to 0.4% annual cumulative frequency of occurrence (warm conditions)

        Args:
            value (float): value for IDD Field `wb_db004`
                Unit: C
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
                                 'for field `wb_db004`'.format(value))

        self._wb_db004 = value