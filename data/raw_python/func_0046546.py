def get_relationship(self, relationship_id):
        """Gets the ``Relationship`` specified by its ``Id``.

        arg:    relationship_id (osid.id.Id): the ``Id`` of the
                ``Relationship`` to retrieve
        return: (osid.relationship.Relationship) - the returned
                ``Relationship``
        raise:  NotFound - no ``Relationship`` found with the given
                ``Id``
        raise:  NullArgument - ``relationship_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('relationship',
                                         collection='Relationship',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(relationship_id, 'relationship').get_identifier())},
                 **self._view_filter()))
        return objects.Relationship(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)