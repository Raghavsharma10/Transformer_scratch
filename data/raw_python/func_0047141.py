def get_objectives_by_genus_type(self, objective_genus_type):
        """Gets an ``ObjectiveList`` corresponding to the given objective genus ``Type`` which does not include objectives of genus types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known objectives
        or an error results. Otherwise, the returned list may contain
        only those objectives that are accessible through this session.

        arg:    objective_genus_type (osid.type.Type): an objective
                genus type
        return: (osid.learning.ObjectiveList) - the returned
                ``Objective`` list
        raise:  NullArgument - ``objective_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('learning',
                                         collection='Objective',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(objective_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.ObjectiveList(result, runtime=self._runtime, proxy=self._proxy)