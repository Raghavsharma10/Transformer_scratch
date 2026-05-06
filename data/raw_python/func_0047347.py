def delete_hierarchy(self, hierarchy_id):
        """Deletes a ``Hierarchy``.

        arg:    hierarchy_id (osid.id.Id): the ``Id`` of the
                ``Hierarchy`` to remove
        raise:  NotFound - ``hierarchy_id`` not found
        raise:  NullArgument - ``hierarchy_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('hierarchy',
                                         collection='Hierarchy',
                                         runtime=self._runtime)
        if not isinstance(hierarchy_id, ABCId):
            return InvalidArgument('the argument is not a valid OSID Id')

        # Should we delete the underlying Relationship Family here???

        collection.delete_one({'_id': ObjectId(hierarchy_id.get_identifier())})