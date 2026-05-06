def get_activity_form_for_create(self, objective_id=None, activity_record_types=None):
        """Gets the activity form for creating new activities.
        A new form should be requested for each create transaction.
        arg:    activityRecordTypes (osid.type.Type): array of activity
                record types
        return: (osid.learning.ActivityForm) - the activity form
        raise:  NotFound - objectiveId is not found
        raise:  NullArgument - objectiveId or activityRecordTypes is
                null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types
        compliance: mandatory - This method must be implemented.

        """
        if objective_id is None:
            raise NullArgument()
        if activity_record_types is None:
            pass  # Still need to deal with the record_types argument
        activity_form = objects.ActivityForm(osid_object_map=None, objective_id=objective_id)
        self._forms[activity_form.get_id().get_identifier()] = not CREATED
        return activity_form