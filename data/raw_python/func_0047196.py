def get_proficiencies_for_resources(self, resource_ids):
        """Gets a ``ProficiencyList`` relating to the given resources.

        arg:    resource_ids (osid.id.IdList): the resource ``Ids``
        return: (osid.learning.ProficiencyList) - the returned
                ``Proficiency`` list
        raise:  NullArgument - ``resource_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_for_source
        # NOTE: This implementation currently ignores plenary and effective views
        collection = JSONClientValidated('learning',
                                         collection='Proficiency',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'resourceId': str(resource_ids)},
                 **self._view_filter())).sort('_sort_id', ASCENDING)
        return objects.ProficiencyList(result, runtime=self._runtime)