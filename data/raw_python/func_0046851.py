def can_update_objective_banks(self, objective_bank_id=None):  # This should not have objective_bank_id argument!
        """Tests if this user can update ObjectiveBanks.
        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known updating an
        ObjectiveBank will result in a PermissionDenied. This is
        intended as a hint to an application that may not wish to offer
        update operations to unauthorized users.
        return: (boolean) - false if ObjectiveBank modification is not
                authorized, true otherwise
        compliance: mandatory - This method must be implemented.

        """
        if not objective_bank_id:
            url_path = construct_url('authorization')
        else:
            url_path = construct_url('authorization',
                                     bank_id=str(objective_bank_id))
        return self._get_request(url_path)['objectiveBankHints']['canUpdate']