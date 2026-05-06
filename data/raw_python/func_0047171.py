def get_activities_for_objective(self, objective_id):
        """Gets the activities for the given objective.

        In plenary mode, the returned list contains all of the
        activities mapped to the objective ``Id`` or an error results if
        an Id in the supplied list is not found or inaccessible.
        Otherwise, inaccessible ``Activities`` may be omitted from the
        list and may present the elements in any order including
        returning a unique set.

        arg:    objective_id (osid.id.Id): ``Id`` of the ``Objective``
        return: (osid.learning.ActivityList) - list of enrollments
        raise:  NotFound - ``objective_id`` not found
        raise:  NullArgument - ``objective_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ActivityLookupSession.get_activities_for_objective_template
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('learning',
                                         collection='Activity',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'objectiveId': str(objective_id)},
                 **self._view_filter()))
        return objects.ActivityList(result, runtime=self._runtime)