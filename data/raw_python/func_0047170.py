def get_activities_by_genus_type(self, activity_genus_type):
        """Gets an ``ActivityList`` corresponding to the given activity genus ``Type`` which does not include activities of genus types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known activities
        or an error results. Otherwise, the returned list may contain
        only those activities that are accessible through this session.

        arg:    activity_genus_type (osid.type.Type): an activity genus
                type
        return: (osid.learning.ActivityList) - the returned ``Activity``
                list
        raise:  NullArgument - ``activity_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('learning',
                                         collection='Activity',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(activity_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.ActivityList(result, runtime=self._runtime, proxy=self._proxy)