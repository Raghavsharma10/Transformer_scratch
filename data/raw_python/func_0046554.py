def get_relationships(self):
        """Gets all ``Relationships``.

        return: (osid.relationship.RelationshipList) - a list of
                ``Relationships``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('relationship',
                                         collection='Relationship',
                                         runtime=self._runtime)
        result = collection.find(self._view_filter()).sort('_id', DESCENDING)
        return objects.RelationshipList(result, runtime=self._runtime, proxy=self._proxy)