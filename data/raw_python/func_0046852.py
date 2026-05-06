def get_objective_bank_form_for_update(self, objective_bank_id=None):
        """Gets the objective bank form for updating an existing objective
        bank.
        A new objective bank form should be requested for each update
        transaction.
        arg:    objectiveBankId (osid.id.Id): the Id of the
                ObjectiveBank
        return: (osid.learning.ObjectiveBankForm) - the objective bank
                form
        raise:  NotFound - objectiveBankId is not found
        raise:  NullArgument - objectiveBankId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_bank_id is None:
            raise NullArgument()
        try:
            url_path = construct_url('objective_banks',
                                     bank_id=objective_bank_id)
            objective_bank = objects.ObjectiveBank(self._get_request(url_path))
        except Exception:
            raise
        objective_bank_form = objects.ObjectiveBankForm(objective_bank._my_map)
        self._forms[objective_bank_form.get_id().get_identifier()] = not UPDATED
        return objective_bank_form