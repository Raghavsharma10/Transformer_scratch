def get_authorizations(self):
        """Geta all ``Authorizations``.

        In plenary mode, the returned list contains all known
        authorizations or an error results. Otherwise, the returned list
        may contain only those authorizations that are accessible
        through this session.

        return: (osid.authorization.AuthorizationList) - a list of
                ``Authorizations``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('authorization',
                                         collection='Authorization',
                                         runtime=self._runtime)
        result = collection.find(self._view_filter()).sort('_id', DESCENDING)
        return objects.AuthorizationList(result, runtime=self._runtime, proxy=self._proxy)