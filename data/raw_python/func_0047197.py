def get_proficiencies_for_objective_and_resource(self, objective_id, resource_id):
        """Gets a ``ProficiencyList`` relating to the given objective and resource ````.

        arg:    objective_id (osid.id.Id): an objective ``Id``
        arg:    resource_id (osid.id.Id): a resource ``Id``
        return: (osid.learning.ProficiencyList) - the returned
                ``Proficiency`` list
        raise:  NullArgument - ``objective_id`` or ``resource_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_for_peers
        # NOTE: This implementation currently ignores plenary and effective views
        collection = JSONClientValidated('learning',
                                         collection='Proficiency',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'resourceId': str(objective_id),
                  'objectiveId': str(resource_id)},
                 **self._view_filter())).sort('_id', ASCENDING)
        return objects.ProficiencyList(result, runtime=self._runtime)