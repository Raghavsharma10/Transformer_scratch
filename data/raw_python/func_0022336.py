def aerosol_optical_depth(self, value=0.999):
        """Corresponds to IDD Field `aerosol_optical_depth`

        Args:
            value (float): value for IDD Field `aerosol_optical_depth`
                Unit: thousandths
                Missing value: 0.999
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = float(value)
            except ValueError:
                raise ValueError(
                    'value {} need to be of type float '
                    'for field `aerosol_optical_depth`'.format(value))

        self._aerosol_optical_depth = value