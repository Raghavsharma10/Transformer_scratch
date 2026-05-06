def get_proficiencies_for_objectives(self, objective_ids):
        """Gets a ``ProficiencyList`` relating to the given objectives.

        arg:    objective_ids (osid.id.IdList): the objective ``Ids``
        return: (osid.learning.ProficiencyList) - the returned
                ``Proficiency`` list
        raise:  NullArgument - ``objective_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_for_destination
        # NOTE: This implementation currently ignores plenary and effective views
        collection = JSONClientValidated('learning',
                                         collection='Proficiency',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'objectiveId': str(objective_ids)},
                 **self._view_filter())).sort('_id', ASCENDING)
        return objects.ProficiencyList(result, runtime=self._runtime)