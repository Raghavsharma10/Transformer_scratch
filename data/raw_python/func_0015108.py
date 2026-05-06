def manage_recurring_payments_profile_status(self, profileid, action,
                                                 note=None):
        """Shortcut to the ManageRecurringPaymentsProfileStatus method.

        ``profileid`` is the same profile id used for getting profile details.
        ``action`` should be either 'Cancel', 'Suspend', or 'Reactivate'.
        ``note`` is optional and is visible to the user. It contains the
            reason for the change in status.
        """
        args = self._sanitize_locals(locals())
        if not note:
            del args['note']
        return self._call('ManageRecurringPaymentsProfileStatus', **args)