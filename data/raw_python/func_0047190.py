def get_proficiencies_by_genus_type(self, proficiency_genus_type):
        """Gets a ``ProficiencyList`` corresponding to the given proficiency genus ``Type`` which does not include proficiencies of types derived from the specified ``Type``.

        arg:    proficiency_genus_type (osid.type.Type): a proficiency
                genus type
        return: (osid.learning.ProficiencyList) - the returned
                ``Proficiency`` list
        raise:  NullArgument - ``proficiency_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('learning',
                                         collection='Proficiency',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(proficiency_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.ProficiencyList(result, runtime=self._runtime, proxy=self._proxy)