def get_activity_form_for_update(self, activity_id=None):
        """Gets the activity form for updating an existing activity.
        A new activity form should be requested for each update
        transaction.
        arg:    activityId (osid.id.Id): the Id of the Activity
        return: (osid.learning.ActivityForm) - the activity form
        raise:  NotFound - activityId is not found
        raise:  NullArgument - activityId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if activity_id is None:
            raise NullArgument()
        try:
            url_path = construct_url('activities',
                                     bank_id=self._catalog_idstr,
                                     act_id=activity_id)
            activity = objects.Activity(self._get_request(url_path))
        except Exception:
            raise
        activity_form = objects.ActivityForm(activity._my_map)
        self._forms[activity_form.get_id().get_identifier()] = not UPDATED
        return activity_form