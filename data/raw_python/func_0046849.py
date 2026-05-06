def get_objective_bank_form_for_create(self, objective_bank_record_types=None):
        """Gets the objective bank form for creating new objective banks.
        A new form should be requested for each create transaction.
        arg:    objectiveBankRecordTypes (osid.type.Type): array of
                objective bank record types
        return: (osid.learning.ObjectiveBankForm) - the objective bank
                form
        raise:  NullArgument - objectiveBankRecordTypes is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types.
        compliance: mandatory - This method must be implemented.

        """
        if objective_bank_record_types is None:
            pass  # Still need to deal with the record_types argument
        objective_bank_form = objects.ObjectiveBankForm()
        self._forms[objective_bank_form.get_id().get_identifier()] = not CREATED
        return objective_bank_form