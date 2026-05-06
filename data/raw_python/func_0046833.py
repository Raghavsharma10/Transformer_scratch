def get_activities_for_objective(self, objective_id=None):
        """Gets the activities for the given objective.
        In plenary mode, the returned list contains all of the
        activities mapped to the objective Id or an error results if an
        Id in the supplied list is not found or inaccessible. Otherwise,
        inaccessible Activities may be omitted from the list and may
        present the elements in any order including returning a unique
        set.
        arg:    objectiveId (osid.id.Id): Id of the Objective
        return: (osid.learning.ActivityList) - list of enrollments
        raise:  NotFound - objectiveId not found
        raise:  NullArgument - objectiveId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method is must be implemented.

        """
        if objective_id is None:
            raise NullArgument()
        # Should also check if objective_id exists?
        url_path = construct_url('activities',
                                 bank_id=self._catalog_idstr,
                                 obj_id=objective_id)
        return objects.ActivityList(self._get_request(url_path))