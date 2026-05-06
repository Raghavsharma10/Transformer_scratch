def get_activity(self, activity_id=None):
        """Gets the Activity specified by its Id.
        In plenary mode, the exact Id is found or a NotFound results.
        Otherwise, the returned Activity may have a different Id than
        requested, such as the case where a duplicate Id was assigned to
        a Activity and retained for compatibility.
        arg:    activityId (osid.id.Id): Id of the Activity
        return: (osid.learning.Activity) - the activity
        raise:  NotFound - activityId not found
        raise:  NullArgument - activityId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method is must be implemented.

        """
        if activity_id is None:
            raise NullArgument()
        url_path = construct_url('activities',
                                 bank_id=self._catalog_idstr,
                                 act_id=activity_id)
        return objects.Activity(self._get_request(url_path))