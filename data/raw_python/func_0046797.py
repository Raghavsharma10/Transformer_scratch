def get_objective_bank(self):
        """Gets the ObjectiveBank associated with this session.

        return: (osid.learning.ObjectiveBank) - the ObjectiveBank
                associated with this session
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        # This should probably be accomplished via a handcar call instead of OSID
        url_path = construct_url('objective_banks',
                                 bank_id=self._catalog_idstr)
        return objects.ObjectiveBank(self._get_request(url_path))