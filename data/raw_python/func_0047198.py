def get_proficiencies(self):
        """Gets all ``Proficiencies``.

        return: (osid.learning.ProficiencyList) - a list of
                ``Proficiencies``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('learning',
                                         collection='Proficiency',
                                         runtime=self._runtime)
        result = collection.find(self._view_filter()).sort('_id', DESCENDING)
        return objects.ProficiencyList(result, runtime=self._runtime, proxy=self._proxy)