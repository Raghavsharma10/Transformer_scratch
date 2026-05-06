def get_proficiency(self, proficiency_id):
        """Gets the ``Proficiency`` specified by its ``Id``.

        arg:    proficiency_id (osid.id.Id): the ``Id`` of the
                ``Proficiency`` to retrieve
        return: (osid.learning.Proficiency) - the returned
                ``Proficiency``
        raise:  NotFound - no ``Proficiency`` found with the given
                ``Id``
        raise:  NullArgument - ``proficiency_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('learning',
                                         collection='Proficiency',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(proficiency_id, 'learning').get_identifier())},
                 **self._view_filter()))
        return objects.Proficiency(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)