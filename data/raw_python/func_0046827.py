def assign_objective_requisite(self, objective_id=None, requisite_objective_id=None):
        """Creates a requirement dependency between two Objectives.

        arg:    objective_id (osid.id.Id): the Id of the dependent
                Objective
        arg:    requisite_objective_id (osid.id.Id): the Id of the
                required Objective
        raise:  AlreadyExists - objective_id already mapped to
                requisite_objective_id
        raise:  NotFound - objective_id or requisite_objective_id not
                found
        raise:  NullArgument - objective_id or requisite_objective_id is
                null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_id is None or requisite_objective_id is None:
            raise NullArgument()
        ors = ObjectiveRequisiteSession(self._objective_bank_id, runtime=self._runtime)
        ids_arg = {'ids': []}
        for objective in ors.get_requisite_objectives(objective_id):
            if objective.get_id() == requisite_objective_id:
                raise AlreadyExists()
            ids_arg['ids'].append(str(objective.get_id()))
        ids_arg['ids'].append(str(requisite_objective_id))

        url_path = construct_url('requisiteids',
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