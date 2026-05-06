def get_objective(self, objective_id=None):
        """Gets the Objective specified by its Id.
        In plenary mode, the exact Id is found or a NotFound results.
        Otherwise, the returned Objective may have a different Id than
        requested, such as the case where a duplicate Id was assigned to
        an Objective and retained for compatibility.
        arg:    objectiveId (osid.id.Id): Id of the Objective
        return: (osid.learning.Objective) - the objective
        raise:  NotFound - objectiveId not found
        raise:  NullArgument - objectiveId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method is must be implemented.

        """
        if objective_id is None:
            raise NullArgument()
        url_path = construct_url('objectives', obj_id=objective_id)
        return objects.Objective(self._get_request(url_path))