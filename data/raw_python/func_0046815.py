def get_child_objective_ids(self, objective_id=None):
        """Gets the child Ids of the given objective.

        arg:    objective_id (osid.id.Id): the Id to query
        return: (osid.id.IdList) - the children of the objective
        raise:  NotFound - objective_id is not found
        raise:  NullArgument - objective_id is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_id is None:
            raise NullArgument()
        url_path = construct_url('childids',
                                 bank_id=self._catalog_idstr,
                                 obj_id=objective_id)
        id_list = list()
        for identifier in self._get_request(url_path)['ids']:
            id_list.append(Id(idstr=identifier))
        return id_objects.IdList(id_list)