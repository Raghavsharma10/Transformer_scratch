def get_resource(self, resource_id):
        """Gets the ``Resource`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``Resource`` may have a
        different ``Id`` than requested, such as the case where a
        duplicate ``Id`` was assigned to a ``Resource`` and retained for
        compatibility.

        arg:    resource_id (osid.id.Id): the ``Id`` of the ``Resource``
                to retrieve
        return: (osid.resource.Resource) - the returned ``Resource``
        raise:  NotFound - no ``Resource`` found with the given ``Id``
        raise:  NullArgument - ``resource_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(resource_id, 'resource').get_identifier())},
                 **self._view_filter()))
        return objects.Resource(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)