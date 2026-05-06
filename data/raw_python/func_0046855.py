def can_delete_objective_banks(self, objective_bank_id=None):  # This should not have objective_bank_id argument!
        """Tests if this user can delete objective banks.
        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known deleting an
        ObjectiveBank will result in a PermissionDenied. This is
        intended as a hint to an application that may not wish to offer
        delete operations to unauthorized users.
        return: (boolean) - false if ObjectiveBank deletion is not
                authorized, true otherwise
        compliance: mandatory - This method must be implemented.

        """
        if not objective_bank_id:
            url_path = construct_url('authorization')
        else:
            url_path = construct_url('authorization',
                                     bank_id=str(objective_bank_id))
        return self._get_request(url_path)['objectiveBankHints']['canDelete']