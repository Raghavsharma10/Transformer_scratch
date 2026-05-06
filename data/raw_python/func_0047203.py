def delete_proficiency(self, proficiency_id):
        """Deletes a ``Proficiency``.

        arg:    proficiency_id (osid.id.Id): the ``Id`` of the
                ``Proficiency`` to remove
        raise:  NotFound - ``proficiency_id`` not found
        raise:  NullArgument - ``proficiency_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.delete_resource_template
        collection = JSONClientValidated('learning',
                                         collection='Proficiency',
                                         runtime=self._runtime)
        if not isinstance(proficiency_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        proficiency_map = collection.find_one(
            dict({'_id': ObjectId(proficiency_id.get_identifier())},
                 **self._view_filter()))

        objects.Proficiency(osid_object_map=proficiency_map, runtime=self._runtime, proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(proficiency_id.get_identifier())})