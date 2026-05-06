def get_activity(self, activity_id):
        """Gets the ``Activity`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``Activity`` may have a
        different ``Id`` than requested, such as the case where a
        duplicate ``Id`` was assigned to a ``Activity`` and retained for
        compatibility.

        arg:    activity_id (osid.id.Id): ``Id`` of the ``Activity``
        return: (osid.learning.Activity) - the activity
        raise:  NotFound - ``activity_id`` not found
        raise:  NullArgument - ``activity_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('learning',
                                         collection='Activity',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(activity_id, 'learning').get_identifier())},
                 **self._view_filter()))
        return objects.Activity(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)