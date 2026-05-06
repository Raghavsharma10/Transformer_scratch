def is_objective_required(self, objective_id=None, required_objective_id=None):
        """Tests if an objective is required before proceeding with an
        objective.

        arg:    objective_id (osid.id.Id): Id of the dependent Objective
        arg:    required_objective_id (osid.id.Id): Id of the required
                Objective
        return: (boolean) - true if objective_id depends on
                required_objective_id, false otherwise
        raise:  NotFound - objective_id not found
        raise:  NullArgument - objective_id is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_id is None or required_objective_id is None:
            raise NullArgument()
        requisite_objective_ids = list()
        for requisite in self.get_all_requisite_objectives(objective_id):
            requisite_objective_ids.append(requisite.get_id())
        return required_objective_id in requisite_objective_ids