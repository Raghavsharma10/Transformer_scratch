def add_child_objective(self, objective_id=None, child_id=None):
        """Adds a child to an objective.

        arg:    objective_id (osid.id.Id): the Id of an objective
        arg:    child_id (osid.id.Id): the Id of the new child
        raise:  AlreadyExists - objective_id is already a parent of
                child_id
        raise:  NotFound - objective_id or child_id not found
        raise:  NullArgument - objective_id or child_id is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_id is None or child_id is None:
            raise NullArgument()
        ohs = ObjectiveHierarchySession(self._objective_bank_id,
                                        runtime=self._runtime)
        if ohs.is_child_of_objective(child_id, objective_id):
            raise AlreadyExists()
        ids_arg = {'ids': []}
        for ident in ohs.get_child_objective_ids(objective_id):
            ids_arg['ids'].append(str(ident))
        ids_arg['ids'].append(str(child_id))

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