def get_objective_form_for_update(self, objective_id=None):
        """Gets the objective form for updating an existing objective.
        A new objective form should be requested for each update
        transaction.
        arg:    objectiveId (osid.id.Id): the Id of the Objective
        return: (osid.learning.ObjectiveForm) - the objective form
        raise:  NotFound - objectiveId is not found
        raise:  NullArgument - objectiveId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_id is None:
            raise NullArgument()
        try:
            url_path = construct_url('objectives',
                                     bank_id=self._catalog_idstr,
                                     obj_id=objective_id)
            objective = objects.Objective(self._get_request(url_path))
        except Exception:
            raise
        objective_form = objects.ObjectiveForm(objective._my_map)
        self._forms[objective_form.get_id().get_identifier()] = not UPDATED
        return objective_form