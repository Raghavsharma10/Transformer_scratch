def move_objective_behind(self, parent_objective_id=None, reference_objective_id=None, objective_id=None):
        """Moves an objective behind a refrence objective under the given
        parent.

        arg:    parent_objective_id (osid.id.Id): the Id of the parent
                objective
        arg:    reference_objective_id (osid.id.Id): the Id of the
                objective
        arg:    objective_id (osid.id.Id): the Id of the objective to
                move behind reference_objective_id
        raise:  NotFound - parent_objective_id, reference_objective_id,
                or objective_id not found, or reference_objective_id or
                objective_id is not a child of parent_objective_id
        raise:  NullArgument - parent_objective_id,
                reference_objective_id, or id is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        # NOT YET TESTED:
        if (parent_objective_id is None or
                reference_objective_id is None or
                objective_id is None):
            raise NullArgument()
        ohs = ObjectiveHierarchySession(self._objective_bank_id, runtime=self._runtime)
        if (not ohs.is_child_of_objective(objective_id, parent_objective_id) or
                not ohs.is_child_of_objective(reference_objective_id, parent_objective_id)):
            raise NotFound('The parent objective identified is not the parent of one or both of the other objectives')
        ids_arg = {'ids': []}
        for ident in ohs.get_child_objective_ids(parent_objective_id):
            ids_arg['ids'].append(str(ident))
        if objective_id != reference_objective_id:
            ids_arg['ids'].remove(str(objective_id))
            index = ids_arg['ids'].index(str(reference_objective_id))
            ids_arg['ids'].insert(index + 1, str(objective_id))

        url_path = construct_url('childids',
                                 bank_id=self._catalog_idstr,
                                 obj_id=parent_objective_id)
        try:
            result = self._put_request(url_path, ids_arg)
        except Exception:
            raise

        # The following is not required by the osid specification:
        id_list = list()
        for identifier in result['ids']:
            id_list.append(Id(idstr=identifier))
        return id_objects.IdList(id_list)