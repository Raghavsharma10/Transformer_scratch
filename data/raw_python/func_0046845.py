def get_objective_bank(self, objective_bank_id=None):
        """Gets the ObjectiveBank specified by its Id.
        In plenary mode, the exact Id is found or a NotFound results.
        Otherwise, the returned ObjectiveBank may have a different Id
        than requested, such as the case where a duplicate Id was
        assigned to a ObjectiveBank and retained for compatility.
        arg:    objectiveBankId (osid.id.Id): Id of the ObjectiveBank
        return: (osid.learning.ObjectiveBank) - the objective bank
        raise:  NotFound - objectiveBankId not found
        raise:  NullArgument - objectiveBankId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method is must be implemented.

        """
        if objective_bank_id is None:
            raise NullArgument()
        url_path = construct_url('objective_banks',
                                 bank_id=objective_bank_id)
        return objects.ObjectiveBank(self._get_request(url_path))