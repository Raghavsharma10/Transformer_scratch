def delete_authorization(self, authorization_id):
        """Deletes the ``Authorization`` identified by the given ``Id``.

        arg:    authorization_id (osid.id.Id): the ``Id`` of the
                ``Authorization`` to delete
        raise:  NotFound - an ``Authorization`` was not found identified
                by the given ``Id``
        raise:  NullArgument - ``authorization_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.delete_resource_template
        collection = JSONClientValidated('authorization',
                                         collection='Authorization',
                                         runtime=self._runtime)
        if not isinstance(authorization_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        authorization_map = collection.find_one(
            dict({'_id': ObjectId(authorization_id.get_identifier())},
                 **self._view_filter()))

        objects.Authorization(osid_object_map=authorization_map, runtime=self._runtime, proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(authorization_id.get_identifier())})