def get_resources(self):
        """Gets all ``Resources``.

        In plenary mode, the returned list contains all known resources
        or an error results. Otherwise, the returned list may contain
        only those resources that are accessible through this session.

        return: (osid.resource.ResourceList) - a list of ``Resources``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        result = collection.find(self._view_filter()).sort('_id', DESCENDING)
        return objects.ResourceList(result, runtime=self._runtime, proxy=self._proxy)