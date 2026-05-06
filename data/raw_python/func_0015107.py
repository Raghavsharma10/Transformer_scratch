def get_recurring_payments_profile_details(self, profileid):
        """Shortcut for the GetRecurringPaymentsProfile method.

        This returns details for a recurring payment plan. The ``profileid`` is
        a value included in the response retrieved by the function
        ``create_recurring_payments_profile``. The profile details include the
        data provided when the profile was created as well as default values
        for ignored fields and some pertinent stastics.

        e.g.:
            response = create_recurring_payments_profile(**profile_info)
            profileid = response.PROFILEID
            details = get_recurring_payments_profile(profileid)

        The response from PayPal is somewhat self-explanatory, but for a
        description of each field, visit the following URI:
        https://www.x.com/docs/DOC-1194
        """
        args = self._sanitize_locals(locals())
        return self._call('GetRecurringPaymentsProfileDetails', **args)