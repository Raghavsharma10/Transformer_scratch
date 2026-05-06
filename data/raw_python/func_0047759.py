def delete_resource(self, resource_id):
        """Deletes a ``Resource``.

        arg:    resource_id (osid.id.Id): the ``Id`` of the ``Resource``
                to remove
        raise:  NotFound - ``resource_id`` not found
        raise:  NullArgument - ``resource_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.delete_resource_template
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        if not isinstance(resource_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        resource_map = collection.find_one(
            dict({'_id': ObjectId(resource_id.get_identifier())},
                 **self._view_filter()))

        objects.Resource(osid_object_map=resource_map, runtime=self._runtime, proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(resource_id.get_identifier())})