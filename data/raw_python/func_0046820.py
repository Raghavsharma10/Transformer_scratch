def remove_child_objectives(self, objective_id=None):
        """Removes all children from an objective.

        arg:    objective_id (osid.id.Id): the Id of an objective
        raise:  NotFound - objective_id not found
        raise:  NullArgument - objective_id is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_id is None:
            raise NullArgument()
        ols = ObjectiveLookupSession(self._objective_bank_id, runtime=self._runtime)
        try:
            ols.get_objective(objective_id)
        except:
            raise  # If no objective, get_objectives will raise NotFound
        ids_arg = {'ids': []}
        url_path = construct_url('childids',
                                 bank_id=self._catalog_idstr,
                                 obj_id=objective_id)
        try:
            result = self._put_request(url_path, ids_arg)
        except Exception:
            raise
        id_list = list()
        for identifier in result['ids']:
            id_list.append(Id(idstr=identifier))
        return id_objects.IdList(id_list)