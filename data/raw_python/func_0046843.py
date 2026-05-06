def delete_activity(self, activity_id=None):
        """Deletes the Activity identified by the given Id.

        arg:    activityId (osid.id.Id): the Id of the Activity to
                delete
        raise:  NotFound - an Activity was not found identified by the
                given Id
        raise:  NullArgument - activityId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if activity_id is None:
            raise NullArgument()
        if not isinstance(activity_id, Id):
            raise InvalidArgument('argument type is not an osid Id')

        url_path = construct_url('activities',
                                 bank_id=self._catalog_idstr,
                                 act_id=activity_id)
        result = self._delete_request(url_path)
        return objects.Activity(result)