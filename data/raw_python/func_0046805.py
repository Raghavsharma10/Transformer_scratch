def get_objective_form_for_create(self, objective_record_types=None):
        """Gets the objective form for creating new objectives.
        A new form should be requested for each create transaction.
        arg:    objectiveRecordTypes (osid.type.Type): array of
                objective record types
        return: (osid.learning.ObjectiveForm) - the objective form
        raise:  NullArgument - objectiveRecordTypes is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types
        compliance: mandatory - This method must be implemented.

        """
        if objective_record_types is None:
            pass  # Still need to deal with the record_types argument
        objective_form = objects.ObjectiveForm()
        self._forms[objective_form.get_id().get_identifier()] = not CREATED
        return objective_form