def delete_relationship(self, relationship_id):
        """Deletes a ``Relationship``.

        arg:    relationship_id (osid.id.Id): the ``Id`` of the
                ``Relationship`` to remove
        raise:  NotFound - ``relationship_id`` not found
        raise:  NullArgument - ``relationship_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.delete_resource_template
        collection = JSONClientValidated('relationship',
                                         collection='Relationship',
                                         runtime=self._runtime)
        if not isinstance(relationship_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        relationship_map = collection.find_one(
            dict({'_id': ObjectId(relationship_id.get_identifier())},
                 **self._view_filter()))

        objects.Relationship(osid_object_map=relationship_map, runtime=self._runtime, proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(relationship_id.get_identifier())})