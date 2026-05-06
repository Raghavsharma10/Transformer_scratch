def leapyear_observed(self, value=None):
        """Corresponds to IDD Field `leapyear_observed` Yes if Leap Year will
        be observed for this file No if Leap Year days (29 Feb) should be
        ignored in this file.

        Args:
            value (str): value for IDD Field `leapyear_observed`
                Accepted values are:
                      - Yes
                      - No
                if `value` is None it will not be checked against the
                specification and is assumed to be a missing value

        Raises:
            ValueError: if `value` is not a valid value

        """
        if value is not None:
            try:
                value = str(value)
            except ValueError:
                raise ValueError('value {} need to be of type str '
                                 'for field `leapyear_observed`'.format(value))
            if ',' in value:
                raise ValueError('value should not contain a comma '
                                 'for field `leapyear_observed`')
            vals = set()
            vals.add("Yes")
            vals.add("No")
            if value not in vals:
                raise ValueError('value {} is not an accepted value for '
                                 'field `leapyear_observed`'.format(value))

        self._leapyear_observed = value