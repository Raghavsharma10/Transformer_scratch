def get_activities_by_genus_type(self, activity_genus_type=None):
        """Gets an ActivityList corresponding to the given activity genus
        Type which does not include activities of genus types derived
        from the specified Type.
        In plenary mode, the returned list contains all known activities
        or an error results. Otherwise, the returned list may contain
        only those activities that are accessible through this session.
        arg:    activityGenusType (osid.type.Type): an activity genus
                type
        return: (osid.learning.ActivityList) - the returned Activity
                list
        raise:  NullArgument - activityGenusType is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if activity_genus_type is None:
            raise NullArgument()
        url_path = construct_url('activities_by_genus',
                                 bank_id=self._catalog_idstr,
                                 genus_type=activity_genus_type.get_identifier())
        return objects.ActivityList(self._get_request(url_path))