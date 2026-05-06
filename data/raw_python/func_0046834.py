def get_activities_for_objectives(self, objective_ids=None):
        """Gets the activities for the given objectives.
        In plenary mode, the returned list contains all of the
        activities specified in the objective Id list, in the order of
        the list, including duplicates, or an error results if a course
        offering Id in the supplied list is not found or inaccessible.
        Otherwise, inaccessible Activities may be omitted from the list
        and may present the elements in any order including returning a
        unique set.
        arg:    objectiveIds (osid.id.IdList): list of objective Ids
        return: (osid.learning.ActivityList) - list of activities
        raise:  NotFound - an objectiveId not found
        raise:  NullArgument - objectiveIdList is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method is must be implemented.

        """
        if objective_ids is None:
            raise NullArgument()
        # Should also check if objective_id exists?
        activities = []
        for i in objective_ids:
            acts = None
            url_path = construct_url('activities',
                                     bank_id=self._catalog_idstr,
                                     obj_id=i)
            try:
                acts = json.loads(self._get_request(url_path))
            except (NotFound, OperationFailed):
                if self._activity_view == PLENARY:
                    raise
                else:
                    pass
            if acts:
                activities += acts
        return objects.ActivityList(activities)