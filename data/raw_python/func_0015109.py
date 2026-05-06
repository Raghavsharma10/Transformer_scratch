def update_recurring_payments_profile(self, profileid, **kwargs):
        """Shortcut to the UpdateRecurringPaymentsProfile method.

        ``profileid`` is the same profile id used for getting profile details.

        The keyed arguments are data in the payment profile which you wish to
        change. The profileid does not change. Anything else will take the new
        value. Most of, though not all of, the fields available are shared
        with creating a profile, but for the complete list of parameters, you
        can visit the following URI:
        https://www.x.com/docs/DOC-1212
        """
        kwargs.update(self._sanitize_locals(locals()))
        return self._call('UpdateRecurringPaymentsProfile', **kwargs)