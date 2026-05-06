def get_relationships_by_genus_type(self, relationship_genus_type):
        """Gets a ``RelationshipList`` corresponding to the given relationship genus ``Type`` which does not include relationships of types derived from the specified ``Type``.

        arg:    relationship_genus_type (osid.type.Type): a relationship
                genus type
        return: (osid.relationship.RelationshipList) - the returned
                ``Relationship list``
        raise:  NullArgument - ``relationship_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('relationship',
                                         collection='Relationship',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(relationship_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.RelationshipList(result, runtime=self._runtime, proxy=self._proxy)