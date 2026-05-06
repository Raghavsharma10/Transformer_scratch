def get_authorizations_by_genus_type(self, authorization_genus_type):
        """Gets an ``AuthorizationList`` corresponding to the given authorization genus ``Type`` which does not include authorizations of genus types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known
        authorizations or an error results. Otherwise, the returned list
        may contain only those authorizations that are accessible
        through this session.

        arg:    authorization_genus_type (osid.type.Type): an
                authorization genus type
        return: (osid.authorization.AuthorizationList) - the returned
                ``Authorization`` list
        raise:  NullArgument - ``authorization_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('authorization',
                                         collection='Authorization',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(authorization_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.AuthorizationList(result, runtime=self._runtime, proxy=self._proxy)