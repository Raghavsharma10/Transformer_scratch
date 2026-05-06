def get_child_objectives(self, objective_id=None):
        """Gets the children of the given objective.

        arg:    objective_id (osid.id.Id): the Id to query
        return: (osid.learning.ObjectiveList) - the children of the
                objective
        raise:  NotFound - objective_id is not found
        raise:  NullArgument - objective_id is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_id is None:
            raise NullArgument()
        url_path = construct_url('children',
                                 bank_id=self._catalog_idstr,
                                 obj_id=objective_id)
        return objects.ObjectiveList(self._get_request(url_path))