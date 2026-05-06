def is_parent_of_objective(self, id_=None, objective_id=None):
        """Tests if an Id is a direct parent of an objective.

        arg:    id (osid.id.Id): an Id
        arg:    objective_id (osid.id.Id): the Id of an objective
        return: (boolean) - true if this id is a parent of objective_id,
                false otherwise
        raise:  NotFound - objective_id is not found
        raise:  NullArgument - id or objective_id is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.
        implementation notes: If id not found return false.

        """
        if id_ is None or objective_id is None:
            raise NullArgument()
        return id_ in list(self.get_parent_objective_ids(objective_id))