def get_authorization(self, authorization_id):
        """Gets the ``Authorization`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``Authorization`` may have a
        different ``Id`` than requested, such as the case where a
        duplicate ``Id`` was assigned to an ``Authorization`` and
        retained for compatibility.

        arg:    authorization_id (osid.id.Id): the ``Id`` of the
                ``Authorization`` to retrieve
        return: (osid.authorization.Authorization) - the returned
                ``Authorization``
        raise:  NotFound - no ``Authorization`` found with the given
                ``Id``
        raise:  NullArgument - ``authorization_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('authorization',
                                         collection='Authorization',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(authorization_id, 'authorization').get_identifier())},
                 **self._view_filter()))
        return objects.Authorization(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)