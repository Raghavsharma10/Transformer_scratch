def get_authorizations_for_resource_and_function(self, resource_id, function_id):
        """Gets a list of ``Authorizations`` associated with a given resource.

        Authorizations related to the given resource, including those
        related through an ``Agent,`` are returned. In plenary mode, the
        returned list contains all known authorizations or an error
        results. Otherwise, the returned list may contain only those
        authorizations that are accessible through this session.

        arg:    resource_id (osid.id.Id): a resource ``Id``
        arg:    function_id (osid.id.Id): a function ``Id``
        return: (osid.authorization.AuthorizationList) - the returned
                ``Authorization list``
        raise:  NullArgument - ``resource_id`` or ``function_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_for_peers
        # NOTE: This implementation currently ignores plenary and effective views
        collection = JSONClientValidated('authorization',
                                         collection='Authorization',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'sourceId': str(resource_id),
                  'destinationId': str(function_id)},
                 **self._view_filter())).sort('_id', ASCENDING)
        return objects.AuthorizationList(result, runtime=self._runtime)