def delete_objective(self, objective_id=None):
        """Deletes the Objective identified by the given Id.

        arg:    objectiveId (osid.id.Id): the Id of the Objective to
                delete
        raise:  NotFound - an Objective was not found identified by the
                given Id
        raise:  NullArgument - objectiveId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_id is None:
            raise NullArgument()
        if not isinstance(objective_id, abc_id):
            raise InvalidArgument('argument type is not an osid Id')

        url_path = construct_url('objectives',
                                 bank_id=self._catalog_idstr,
                                 obj_id=objective_id)
        result = self._delete_request(url_path)
        return objects.Objective(result)