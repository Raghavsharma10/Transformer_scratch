def get_activities_by_ids(self, activity_ids=None):
        """Gets an ActivityList corresponding to the given IdList.
        In plenary mode, the returned list contains all of the
        activities specified in the Id list, in the order of the list,
        including duplicates, or an error results if an Id in the
        supplied list is not found or inaccessible. Otherwise,
        inaccessible Activities may be omitted from the list and may
        present the elements in any order including returning a unique
        set.
        arg:    activityIds (osid.id.IdList): the list of Ids to
                retrieve
        return: (osid.learning.ActivityList) - the returned Activity
                list
        raise:  NotFound - an Id was not found
        raise:  NullArgument - activityIds is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if activity_ids is None:
            raise NullArgument()
        activities = []
        for i in activity_ids:
            activity = None
            url_path = construct_url('activities',
                                     bank_id=self._catalog_idstr,
                                     act_id=i)
            try:
                activity = self._get_request(url_path)
            except (NotFound, OperationFailed):
                if self._activity_view == PLENARY:
                    raise
                else:
                    pass
            if activity:
                if not (self._activity_view == COMPARATIVE and
                        activity in activities):
                    activities.append(activity)
        return objects.ActivityList(activities)