def can_lookup_objective_banks(self):
        """Tests if this user can perform ObjectiveBank lookups.
        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known all methods in this
        session will result in a PermissionDenied. This is intended as
        a hint to an application that may opt not to offer lookup
        operations to unauthorized users.
        return: (boolean) - false if lookup methods are not authorized,
                true otherwise
        compliance: mandatory - This method must be implemented.

        """
        # need to use a default bank_id here...not ideal.
        url_path = construct_url('authorization',
                                 bank_id=self._default_bank_id)
        return self._get_request(url_path)['objectiveBankHints']['canLookup']