def get_resources_by_genus_type(self, resource_genus_type):
        """Gets a ``ResourceList`` corresponding to the given resource genus ``Type`` which does not include resources of types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known resources
        or an error results. Otherwise, the returned list may contain
        only those resources that are accessible through this session.

        arg:    resource_genus_type (osid.type.Type): a resource genus
                type
        return: (osid.resource.ResourceList) - the returned ``Resource``
                list
        raise:  NullArgument - ``resource_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(resource_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.ResourceList(result, runtime=self._runtime, proxy=self._proxy)