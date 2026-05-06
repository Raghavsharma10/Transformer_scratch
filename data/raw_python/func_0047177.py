def delete_activity(self, activity_id):
        """Deletes the ``Activity`` identified by the given ``Id``.

        arg:    activity_id (osid.id.Id): the ``Id`` of the ``Activity``
                to delete
        raise:  NotFound - an ``Activity`` was not found identified by
                the given ``Id``
        raise:  NullArgument - ``activity_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.delete_resource_template
        collection = JSONClientValidated('learning',
                                         collection='Activity',
                                         runtime=self._runtime)
        if not isinstance(activity_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        activity_map = collection.find_one(
            dict({'_id': ObjectId(activity_id.get_identifier())},
                 **self._view_filter()))

        objects.Activity(osid_object_map=activity_map, runtime=self._runtime, proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(activity_id.get_identifier())})